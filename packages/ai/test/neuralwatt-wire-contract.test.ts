/**
 * Wire contract for Neuralwatt (https://api.neuralwatt.com/v1). Fixed
 * `/v1/models` rows go through the real discovery mapper and `buildModel`, and
 * requests are encoded through `streamSimple` the way a session sends them:
 * system prompts in the `system` role, efforts and their aliases from each
 * row, output caps up to the model's reported limit, and reasoning sent back
 * on the next turn.
 */
import { beforeAll, describe, expect, it } from "bun:test";
import { streamSimple } from "@oh-my-pi/pi-ai/stream";
import type { AssistantMessage, Context, FetchImpl, Model } from "@oh-my-pi/pi-ai/types";
import { buildModel } from "@oh-my-pi/pi-catalog/build";
import { Effort } from "@oh-my-pi/pi-catalog/effort";
import { clampThinkingLevelForModel } from "@oh-my-pi/pi-catalog/model-thinking";
import { getBundledModel } from "@oh-my-pi/pi-catalog/models";
import { neuralwattModelManagerOptions } from "@oh-my-pi/pi-catalog/provider-models/openai-compat";

interface RowShape {
	efforts: string[];
	defaultEffort: string;
	mandatory?: boolean;
	aliases: Record<string, string>;
	maxOutput: number | null;
	context: number;
}

/** A `/v1/models` row reduced to the fields the mapper reads. */
function nwRow(id: string, shape: RowShape): Record<string, unknown> {
	return {
		id,
		object: "model",
		owned_by: "neuralwatt",
		max_model_len: shape.context,
		metadata: {
			display_name: id,
			pricing: { input_per_million: 1, output_per_million: 2, cached_input_per_million: 0.1, pricing_tbd: false },
			capabilities: { tools: true, vision: true, reasoning: true, developer_role: false },
			reasoning: {
				mandatory: shape.mandatory ?? false,
				supported_efforts: shape.efforts,
				default_effort: shape.defaultEffort,
				effort_aliases: shape.aliases,
			},
			limits: { max_context_length: shape.context, max_output_tokens: shape.maxOutput },
			deprecated: false,
		},
	};
}

const KIMI_K31 = {
	efforts: ["max", "high", "medium", "low", "none"],
	defaultEffort: "max",
	aliases: { xhigh: "max", minimal: "low" },
	maxOutput: null,
	context: 1048560,
};

/** Live reasoning controls and limits as of 2026-11-20. */
const ROWS = [
	nwRow("nw-large", KIMI_K31),
	nwRow("kimi-k3.1-pro", KIMI_K31),
	nwRow("nw-flash", {
		efforts: ["max", "xhigh", "high", "low", "none"],
		defaultEffort: "high",
		aliases: { medium: "high", minimal: "low" },
		maxOutput: 393216,
		context: 1048560,
	}),
	nwRow("nw-small", {
		efforts: ["xhigh", "medium", "low", "none"],
		defaultEffort: "xhigh",
		aliases: { max: "xhigh", high: "xhigh", minimal: "low" },
		maxOutput: 131072,
		context: 262128,
	}),
	nwRow("glm-5.3", {
		efforts: ["max", "high", "low"],
		defaultEffort: "max",
		mandatory: true,
		aliases: { xhigh: "max", medium: "high", minimal: "low" },
		maxOutput: null,
		context: 1048560,
	}),
	nwRow("mimo-v2.6-pro", {
		efforts: ["high", "none"],
		defaultEffort: "high",
		aliases: { max: "high", xhigh: "high", medium: "high", low: "high", minimal: "none" },
		maxOutput: 393216,
		context: 1048560,
	}),
	nwRow("gemma-4-31b", {
		efforts: ["max", "none"],
		defaultEffort: "none",
		aliases: { xhigh: "max", high: "max", medium: "max", low: "max", minimal: "max" },
		maxOutput: 16384,
		context: 262128,
	}),
];

const models = new Map<string, Model<"openai-completions">>();

beforeAll(async () => {
	const fetchRows: FetchImpl = async () => Response.json({ object: "list", data: ROWS });
	const specs = (await neuralwattModelManagerOptions({ fetch: fetchRows }).fetchDynamicModels?.()) ?? [];
	for (const spec of specs) models.set(spec.id, buildModel(spec));
});

function model(id: string): Model<"openai-completions"> {
	const found = models.get(id);
	if (!found) throw new Error(`neuralwatt fixture model ${id} was not built`);
	return found;
}

const USER = { role: "user" as const, content: "What is 2+2?", timestamp: 0 };
const CONTEXT: Context = { systemPrompt: ["You are terse."], messages: [USER] };

function sse(id: string, deltas: Record<string, unknown>[], finishReason: string): Response {
	const chunk = (delta: Record<string, unknown>, extra: Record<string, unknown> = {}) =>
		JSON.stringify({
			id: "chatcmpl-nw",
			object: "chat.completion.chunk",
			model: id,
			choices: [{ index: 0, delta, ...extra }],
		});
	const lines = [...deltas.map(delta => chunk(delta)), chunk({}, { finish_reason: finishReason }), "[DONE]"];
	return new Response(`${lines.map(line => `data: ${line}`).join("\n\n")}\n\n`, {
		status: 200,
		headers: { "content-type": "text/event-stream" },
	});
}

type Options = { reasoning?: Effort; disableReasoning?: boolean; maxTokens?: number };

/** Sends one request and returns its body, with the reply the server streamed. */
async function exchange(
	id: string,
	context: Context,
	options: Options,
	reply: { deltas: Record<string, unknown>[]; finishReason: string } = {
		deltas: [{ content: "4" }],
		finishReason: "stop",
	},
): Promise<{ body: Record<string, unknown>; message: AssistantMessage }> {
	let body: Record<string, unknown> | undefined;
	const fetchMock: FetchImpl = Object.assign(
		async (_input: string | URL | Request, init?: RequestInit): Promise<Response> => {
			body = JSON.parse(typeof init?.body === "string" ? init.body : "{}") as Record<string, unknown>;
			return sse(id, reply.deltas, reply.finishReason);
		},
		{ preconnect: fetch.preconnect },
	);
	const message = await streamSimple(model(id), context, { apiKey: "sk-test", fetch: fetchMock, ...options }).result();
	if (!body) throw new Error("Expected a captured request body");
	return { body, message };
}

async function capture(id: string, options: Options): Promise<Record<string, unknown>> {
	return (await exchange(id, CONTEXT, options)).body;
}

describe("Neuralwatt wire contract", () => {
	it("sends the row's own off and medium for the alias and the id Kimi K3 rules recognize", async () => {
		for (const id of ["nw-large", "kimi-k3.1-pro"]) {
			expect((await capture(id, { disableReasoning: true })).reasoning_effort).toBe("none");
			expect((await capture(id, { reasoning: Effort.Medium })).reasoning_effort).toBe("medium");
		}
		// A mandatory row clamps off to its lowest effort.
		expect((await capture("glm-5.3", { disableReasoning: true })).reasoning_effort).toBe("low");
	});

	it("sends the effort each row's aliases name for efforts it does not serve", async () => {
		// Served efforts stay the selectable ladder.
		expect(model("nw-small").thinking?.efforts).toEqual([Effort.Low, Effort.Medium, Effort.XHigh]);
		// A programmatic request for an unserved effort follows the alias up.
		expect((await capture("nw-small", { reasoning: Effort.High })).reasoning_effort).toBe("xhigh");
		// A configured effort is clamped first, the way a session resolves it.
		const configured = (id: string, effort: Effort) => ({ reasoning: clampThinkingLevelForModel(model(id), effort) });
		expect((await capture("glm-5.3", configured("glm-5.3", Effort.Medium))).reasoning_effort).toBe("high");
		// MiMo aliases `minimal` to off and every other effort to `high`.
		expect((await capture("mimo-v2.6-pro", configured("mimo-v2.6-pro", Effort.Minimal))).reasoning_effort).toBe(
			"none",
		);
		expect((await capture("mimo-v2.6-pro", configured("mimo-v2.6-pro", Effort.Low))).reasoning_effort).toBe("high");
	});

	it("still clamps an alias outside the effort vocabulary that is not off on other transports", async () => {
		// A custom MiniMax-M2 narrowed to `high` keeps its `medium: "adaptive"`
		// map. Keeping `medium` would fail the Anthropic encoder before fetch.
		const bundled = getBundledModel<"anthropic-messages">("minimax", "MiniMax-M2");
		if (!bundled.thinking) throw new Error("bundled MiniMax-M2 lost its thinking config");
		const narrowed: Model<"anthropic-messages"> = {
			...bundled,
			thinking: { ...bundled.thinking, efforts: [Effort.High] },
		};
		const reasoning = clampThinkingLevelForModel(narrowed, Effort.Medium);
		expect(reasoning).toBe(Effort.High);
		let body: Record<string, unknown> | undefined;
		const fetchMock: FetchImpl = Object.assign(
			async (_input: string | URL | Request, init?: RequestInit): Promise<Response> => {
				body = JSON.parse(typeof init?.body === "string" ? init.body : "{}") as Record<string, unknown>;
				return Response.json(
					{ type: "error", error: { type: "invalid_request_error", message: "stop" } },
					{ status: 400 },
				);
			},
			{ preconnect: fetch.preconnect },
		);
		await streamSimple(narrowed, CONTEXT, {
			apiKey: "sk-test",
			fetch: fetchMock,
			reasoning,
			maxHttpAttempts: 1,
		}).result();
		expect(body?.thinking).toMatchObject({ type: "adaptive" });
	});

	it("sends system prompts in the system role and caps output at the model's own limit", async () => {
		const flash = await capture("nw-flash", {});
		expect((flash.messages as { role: string }[])[0].role).toBe("system");
		expect(flash.store).toBeUndefined();
		expect(flash.max_tokens).toBe(393216);
		expect((await capture("nw-flash", { maxTokens: 100_000 })).max_tokens).toBe(100_000);
	});

	it("uses the 131072 fallback as the default and the cap where a row reports no output limit", async () => {
		expect((await capture("nw-large", {})).max_tokens).toBe(131072);
		// omp keeps one output limit per model, so an explicit larger request stops at it.
		expect((await capture("nw-large", { maxTokens: 200_000 })).max_tokens).toBe(131072);
		// A reported limit wins.
		expect((await capture("gemma-4-31b", { maxTokens: 30_000 })).max_tokens).toBe(16384);
	});

	it("sends a tool-call turn's reasoning back in the field the model streamed it in", async () => {
		const toolCall = {
			index: 0,
			id: "call_nw",
			type: "function",
			function: { name: "read", arguments: '{"path":"README.md"}' },
		};
		for (const [id, field] of [
			["nw-large", "reasoning_content"],
			["nw-flash", "reasoning"],
		] as const) {
			const first = await exchange(
				id,
				CONTEXT,
				{ reasoning: Effort.High },
				{ deltas: [{ [field]: "Read the file first." }, { tool_calls: [toolCall] }], finishReason: "tool_calls" },
			);
			expect(first.message.stopReason).toBe("toolUse");
			const toolResult = {
				role: "toolResult" as const,
				toolCallId: "call_nw",
				toolName: "read",
				content: [{ type: "text" as const, text: "# Readme" }],
				isError: false,
				timestamp: 1,
			};
			const followUp = await exchange(
				id,
				{ ...CONTEXT, messages: [USER, first.message, toolResult] },
				{ reasoning: Effort.High },
			);
			const replayed = (followUp.body.messages as Record<string, unknown>[]).find(
				message => message.role === "assistant",
			);
			expect(replayed?.tool_calls).toHaveLength(1);
			expect(replayed?.[field]).toBe("Read the file first.");
		}
	});
});
