/**
 * Neuralwatt request encoding. Fixed `/v1/models` rows go through the real
 * mapper and `buildModel`, then `streamSimple` encodes requests the way a
 * session sends them.
 */
import { beforeAll, describe, expect, it } from "bun:test";
import { streamSimple } from "@oh-my-pi/pi-ai/stream";
import type { AssistantMessage, Context, FetchImpl, Model } from "@oh-my-pi/pi-ai/types";
import { buildModel } from "@oh-my-pi/pi-catalog/build";
import { Effort } from "@oh-my-pi/pi-catalog/effort";
import { clampThinkingLevelForModel } from "@oh-my-pi/pi-catalog/model-thinking";
import { neuralwattModelManagerOptions } from "@oh-my-pi/pi-catalog/provider-models/openai-compat";

/** Live reasoning controls and limits (2026-11-20): supported efforts, aliases, output limit. */
const KIMI: [string[], Record<string, string>, number | null] = [
	["max", "high", "medium", "low", "none"],
	{ xhigh: "max", minimal: "low" },
	null,
];
const SHAPES: Record<string, [string[], Record<string, string>, number | null]> = {
	"nw-large": KIMI,
	"kimi-k3.1-pro": KIMI,
	"nw-flash": [["max", "xhigh", "high", "low", "none"], { medium: "high", minimal: "low" }, 393216],
	"nw-small": [["xhigh", "medium", "low", "none"], { max: "xhigh", high: "xhigh", minimal: "low" }, 131072],
	"glm-5.3": [["max", "high", "low"], { xhigh: "max", medium: "high", minimal: "low" }, null],
	"mimo-v2.6-pro": [
		["high", "none"],
		{ max: "high", xhigh: "high", medium: "high", low: "high", minimal: "none" },
		393216,
	],
	"gemma-4-31b": [["max", "none"], { high: "max", medium: "max", low: "max" }, 16384],
};

const models = new Map<string, Model<"openai-completions">>();

beforeAll(async () => {
	const data = Object.entries(SHAPES).map(([id, [efforts, aliases, maxOutput]]) => ({
		id,
		metadata: {
			pricing: { input_per_million: 1, output_per_million: 2 },
			capabilities: { tools: true, reasoning: true, developer_role: false },
			reasoning: { mandatory: !efforts.includes("none"), supported_efforts: efforts, effort_aliases: aliases },
			limits: { max_context_length: 1048560, max_output_tokens: maxOutput },
		},
	}));
	const fetchRows: FetchImpl = async () => Response.json({ data });
	for (const spec of (await neuralwattModelManagerOptions({ fetch: fetchRows }).fetchDynamicModels?.()) ?? []) {
		models.set(spec.id, buildModel(spec));
	}
});

const model = (id: string) => models.get(id)!;
const USER = { role: "user" as const, content: "What is 2+2?", timestamp: 0 };
const CONTEXT: Context = { systemPrompt: ["You are terse."], messages: [USER] };
type Options = { reasoning?: Effort; disableReasoning?: boolean; maxTokens?: number };

/** Sends one request, answers it with `deltas`, and returns the body and reply. */
async function exchange(id: string, options: Options, context = CONTEXT, deltas: object[] = [{ content: "4" }]) {
	let body: Record<string, any> = {};
	const chunks = [
		...deltas.map(delta => ({ delta })),
		{ delta: {}, finish_reason: deltas.some(delta => "tool_calls" in delta) ? "tool_calls" : "stop" },
	];
	const sse = chunks.map(choice => `data: ${JSON.stringify({ id: "nw", choices: [{ index: 0, ...choice }] })}\n\n`);
	const fetchMock: FetchImpl = Object.assign(
		async (_input: string | URL | Request, init?: RequestInit) => {
			body = JSON.parse(String(init?.body));
			return new Response(`${sse.join("")}data: [DONE]\n\n`, { headers: { "content-type": "text/event-stream" } });
		},
		{ preconnect: fetch.preconnect },
	);
	const message = await streamSimple(model(id), context, { apiKey: "sk-test", fetch: fetchMock, ...options }).result();
	return { body, message };
}
const effortFor = async (id: string, options: Options) => (await exchange(id, options)).body.reasoning_effort;
const configured = (id: string, effort: Effort) => ({ reasoning: clampThinkingLevelForModel(model(id), effort) });

describe("Neuralwatt wire contract", () => {
	it("sends each row's own efforts, off value and aliases", async () => {
		// Kimi K3 class rules must not make the row mandatory or remap medium.
		for (const id of ["nw-large", "kimi-k3.1-pro"]) {
			expect(await effortFor(id, { disableReasoning: true })).toBe("none");
			expect(await effortFor(id, { reasoning: Effort.Medium })).toBe("medium");
		}
		expect(await effortFor("glm-5.3", { disableReasoning: true })).toBe("low");
		expect(model("nw-small").thinking?.efforts).toEqual([Effort.Low, Effort.Medium, Effort.XHigh]);
		expect(await effortFor("nw-small", { reasoning: Effort.High })).toBe("xhigh");
		expect(await effortFor("glm-5.3", configured("glm-5.3", Effort.Medium))).toBe("high");
		expect(await effortFor("mimo-v2.6-pro", configured("mimo-v2.6-pro", Effort.Minimal))).toBe("none");
		expect(await effortFor("mimo-v2.6-pro", configured("mimo-v2.6-pro", Effort.Low))).toBe("high");
	});

	it("sends the system role and caps output at the model's limit or the 131072 fallback", async () => {
		const maxTokens = async (id: string, options: Options = {}) => (await exchange(id, options)).body.max_tokens;
		const { body } = await exchange("nw-flash", {});
		expect(body.messages[0].role).toBe("system");
		expect(body.store).toBeUndefined();
		expect(body.max_tokens).toBe(393216);
		expect(await maxTokens("nw-large")).toBe(131072);
		// omp keeps one output limit per model, so a larger explicit request stops at it.
		expect(await maxTokens("nw-large", { maxTokens: 200_000 })).toBe(131072);
		expect(await maxTokens("gemma-4-31b", { maxTokens: 30_000 })).toBe(16384);
	});

	it("sends a tool-call turn's reasoning back in the field the model streamed it in", async () => {
		const call = { index: 0, id: "call_nw", type: "function", function: { name: "read", arguments: "{}" } };
		for (const [id, field] of [
			["nw-large", "reasoning_content"],
			["nw-flash", "reasoning"],
		] as const) {
			const first: AssistantMessage = (
				await exchange(id, { reasoning: Effort.High }, CONTEXT, [{ [field]: "Read it." }, { tool_calls: [call] }])
			).message;
			const result = { role: "toolResult" as const, toolCallId: "call_nw", toolName: "read", isError: false };
			const messages = [USER, first, { ...result, content: [{ type: "text" as const, text: "ok" }], timestamp: 1 }];
			const { body } = await exchange(id, { reasoning: Effort.High }, { ...CONTEXT, messages });
			const replayed = body.messages.find((message: { role: string }) => message.role === "assistant");
			expect(replayed.tool_calls).toHaveLength(1);
			expect(replayed[field]).toBe("Read it.");
		}
	});
});
