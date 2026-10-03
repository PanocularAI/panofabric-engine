"""A toy lookup-and-compute task: the taskset, its rewards, and the agent loop.

Each task names 3-10 parcels whose weights only the ``get_weight`` tool
reveals, each in its own unit (kg, g or lb). The agent looks each one up,
converts, and reports the total in kilograms through ``submit_total``. Reward
is how close that total is (1 when exact, falling to 0 at a 100% error) minus
0.25 per tool-call error -- the usual shape of scoring an agent on what its tool
calls achieved and on the calls it got wrong.

The agent loop (``ParcelHarness.run_agent``) is plain OpenAI-client code: swap
in your own loop and tools, keep the ``client`` it is handed.
"""

import json
import random

import verifiers.v1 as vf

from panoengine.train.rl.harness import AgentHarness

TOOL_ERROR_PENALTY = 0.25
TO_KG = {"kg": 1.0, "g": 0.001, "lb": 0.4536}

SYSTEM_PROMPT = (
    "You answer questions about parcels. Call get_weight to look up each "
    "parcel's weight; weights come back in kg, g or lb (1 lb = 0.4536 kg). Then "
    "call submit_total exactly once with the total in kilograms. Reply with one "
    "short sentence after submitting."
)

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weight",
            "description": "The weight of one parcel, with its unit.",
            "parameters": {
                "type": "object",
                "properties": {"parcel_id": {"type": "string"}},
                "required": ["parcel_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "submit_total",
            "description": "Submit the final answer: the total weight in kilograms.",
            "parameters": {
                "type": "object",
                "properties": {"total_kg": {"type": "number"}},
                "required": ["total_kg"],
            },
        },
    },
]


class ParcelData(vf.TaskData):
    readings: dict[str, tuple[float, str]]   # parcel id -> (value, unit)

    @property
    def total_kg(self) -> float:
        return sum(value * TO_KG[unit] for value, unit in self.readings.values())


class ParcelTask(vf.Task[ParcelData]):
    data: ParcelData

    @vf.reward(weight=1.0)
    async def accuracy(self, trace: vf.Trace) -> float:
        """Graded, not exact-match. Once a policy solves a task, every rollout
        in a group scores the same, GRPO has nothing to rank, and the run stops
        on "consecutive untrainable batches"; partial credit (and a task with
        real arithmetic in it) keeps groups rankable for longer. Rounded so
        float noise in the last digits does not count as a difference."""
        submitted = trace.info.get("submitted", [])
        if not submitted:
            return 0.0
        truth = self.data.total_kg
        return round(max(0.0, 1.0 - abs(submitted[0] - truth) / truth), 2)

    @vf.reward(weight=1.0)
    async def tool_errors(self, trace: vf.Trace) -> float:
        return -TOOL_ERROR_PENALTY * trace.info.get("tool_errors", 0)


class ParcelTasksetConfig(vf.TasksetConfig):
    num_tasks: int = 512
    seed: int = 0


class ParcelTaskset(vf.Taskset[ParcelTask, ParcelTasksetConfig]):
    config: ParcelTasksetConfig

    def load(self) -> list[ParcelTask]:
        rng = random.Random(self.config.seed)
        tasks = []
        for idx in range(self.config.num_tasks):
            ids = [f"P-{rng.randrange(1000, 9999)}" for _ in range(rng.randint(3, 10))]
            readings = {}
            for pid in ids:
                unit = rng.choice(list(TO_KG))
                readings[pid] = (round(rng.uniform(0.5, 30.0) / TO_KG[unit], 2), unit)
            data = ParcelData(
                idx=idx,
                system_prompt=SYSTEM_PROMPT,
                prompt=f"What is the total weight of parcels {', '.join(ids)} in kilograms?",
                readings=readings,
            )
            tasks.append(ParcelTask(data, self.config.task))
        return tasks


class ParcelHarness(AgentHarness):
    max_turns = 12

    async def run_agent(self, client, model, messages, data, trace) -> None:
        trace.info["submitted"] = []
        trace.info["tool_errors"] = 0
        for _ in range(self.max_turns):
            completion = await client.chat.completions.create(
                model=model, messages=messages, tools=TOOLS
            )
            reply = completion.choices[0].message
            messages.append(reply.model_dump(exclude_none=True))
            if not reply.tool_calls:
                return
            for i, call in enumerate(reply.tool_calls):
                result = self._run_tool(call.function.name, call.function.arguments, data, trace)
                messages.append({
                    "role": "tool",
                    "tool_call_id": call.id or f"call_{i}",
                    "content": json.dumps(result),
                })

    @staticmethod
    def _run_tool(name, arguments, data: ParcelData, trace: vf.Trace) -> dict:
        """Execute one call; anything malformed counts as a tool error."""
        try:
            args = json.loads(arguments) if isinstance(arguments, str) else dict(arguments)
            if name == "get_weight":
                parcel_id = args["parcel_id"]
                if parcel_id not in data.readings:
                    raise ValueError(f"unknown parcel {parcel_id!r}")
                value, unit = data.readings[parcel_id]
                return {"parcel_id": parcel_id, "weight": value, "unit": unit}
            if name == "submit_total":
                if trace.info["submitted"]:
                    raise ValueError("a total was already submitted")
                trace.info["submitted"].append(float(args["total_kg"]))
                return {"status": "recorded"}
            raise ValueError(f"unknown tool {name!r}")
        except (ValueError, KeyError, TypeError, json.JSONDecodeError) as e:
            trace.info["tool_errors"] += 1
            return {"error": str(e)}


# Verifiers loads the Taskset AND the Harness from this module's __all__.
__all__ = ["ParcelTaskset", "ParcelHarness"]
