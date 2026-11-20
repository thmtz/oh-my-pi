/**
 * Wire contract for the bundled Neuralwatt rows (https://api.neuralwatt.com/v1),
 * encoded through `streamSimple` the way a session sends them: system prompts
 * in the `system` role, effort and off taken from each row's own ladder, and
 * output caps up to the model's reported limit.
 */
import { describe, expect, it } from "bun:test";
import { streamSimple } from "@oh-my-pi/pi-ai/stream";
import type { Context, FetchImpl, Model } from "@oh-my-pi/pi-ai/types";
import { Effort } from "@oh-my-pi/pi-catalog/effort";
import { getBundledModels } from "@oh-my-pi/pi-catalog/models";

function bundled(id: string): Model<"openai-completions"> {
	const models = getBundledModels("neuralwatt") as readonly Model<"openai-completions">[];
	const model = models.find(candidate => candidate.id === id);
	if (!model) throw new Error(`bundled neuralwatt model ${id} is missing`);
	return model;
}

const CONTEXT: Context = {
	systemPrompt: ["You are terse."],
	messages: [{ role: "user", content: "What is 2+2?", timestamp: 0 }],
};

function sse(model: Model<"openai-completions">): Response {
	const chunk = (delta: Record<string, unknown>, extra: Record<string, unknown> = {}) =>
		JSON.stringify({
			id: "chatcmpl-nw",
			object: "chat.completion.chunk",
			model: model.id,
			choices: [{ index: 0, delta, ...extra }],
		});
	const body = [chunk({ content: "4" }), chunk({}, { finish_reason: "stop" }), "[DONE]"]
		.map(line => `data: ${line}`)
		.join("\n\n");
	return new Response(`${body}\n\n`, { status: 200, headers: { "content-type": "text/event-stream" } });
}

async function capture(
	id: string,
	options: { reasoning?: Effort; disableReasoning?: boolean; maxTokens?: number },
): Promise<Record<string, unknown>> {
	const model = bundled(id);
	let payload: Record<string, unknown> | undefined;
	const fetchMock: FetchImpl = Object.assign(
		async (_input: string | URL | Request, init?: RequestInit): Promise<Response> => {
			payload = JSON.parse(typeof init?.body === "string" ? init.body : "{}") as Record<string, unknown>;
			return sse(model);
		},
		{ preconnect: fetch.preconnect },
	);
	await streamSimple(model, CONTEXT, { apiKey: "sk-test", fetch: fetchMock, ...options }).result();
	if (!payload) throw new Error("Expected a captured request payload");
	return payload;
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

	it("sends system prompts in the system role and caps output at the model's own limit", async () => {
		const flash = await capture("nw-flash", {});
		expect((flash.messages as { role: string }[])[0].role).toBe("system");
		expect(flash.store).toBeUndefined();
		expect(flash.max_tokens).toBe(393216);
		expect((await capture("nw-flash", { maxTokens: 100_000 })).max_tokens).toBe(100_000);
		expect((await capture("gemma-4-31b", { maxTokens: 30_000 })).max_tokens).toBe(16384);
		// The null-limit fallback is the cap for rows that report none.
		expect((await capture("nw-large", {})).max_tokens).toBe(131072);
	});
});
