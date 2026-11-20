import { describe, expect, test } from "bun:test";
import * as fs from "node:fs/promises";
import * as os from "node:os";
import * as path from "node:path";
import { buildModel } from "@oh-my-pi/pi-catalog/build";
import { Effort } from "@oh-my-pi/pi-catalog/effort";
import { resolveProviderModels } from "@oh-my-pi/pi-catalog/model-manager";
import { neuralwattModelManagerOptions } from "@oh-my-pi/pi-catalog/provider-models/openai-compat";
import type { FetchImpl } from "@oh-my-pi/pi-catalog/types";
import { applyGeneratedModelPolicies } from "../scripts/generated-policies";

/**
 * Rows copied from live `https://api.neuralwatt.com/v1/models` (2026-11-20),
 * descriptions trimmed. They cover the shapes the mapper branches on:
 * - `nw-large`: tracking alias, null `max_output_tokens`, optional reasoning
 *   with a `none` off value
 * - `glm-5.3`: mandatory reasoning, no `none`
 * - `gemma-4-31b`: reported output cap, one effort, off by default
 * - `qwen3.6-35b-fast`: no reasoning
 * - `mammoth-1`: decision model without tool calling
 */
const LIVE_ROWS: Record<string, unknown>[] = [
	{
		id: "glm-5.3",
		object: "model",
		created: 0,
		owned_by: "neuralwatt",
		max_model_len: 1048560,
		metadata: {
			display_name: "GLM 5.3",
			provider: "ZhipuAI",
			huggingface_id: "zai-org/GLM-5.3",
			pricing: {
				input_per_million: 1.45,
				output_per_million: 4.5,
				cached_input_per_million: 0.145,
				cached_output_per_million: null,
				currency: "USD",
				pricing_tbd: false,
				cached_input_is_default: true,
				service_tier: "standard",
				flex_discount_multiplier: null,
			},
			capabilities: {
				tools: true,
				json_mode: true,
				vision: false,
				audio_input: false,
				reasoning: true,
				reasoning_effort: true,
				streaming: true,
				system_role: true,
				developer_role: false,
				hosted_tools: false,
			},
			reasoning: {
				mandatory: true,
				default_enabled: true,
				supported_efforts: ["max", "high", "low"],
				default_effort: "max",
				accepted_efforts: ["max", "xhigh", "high", "medium", "low", "minimal"],
				effort_aliases: {
					xhigh: "max",
					medium: "high",
					minimal: "low",
				},
			},
			limits: {
				max_context_length: 1048560,
				max_output_tokens: null,
				max_images: null,
				max_images_total: null,
				max_audio_seconds: null,
				max_audio_clips: null,
				max_audio_bytes: null,
			},
			deprecated: false,
			deprecated_message: null,
		},
		architecture: {
			modality: "text->text",
			input_modalities: ["text"],
			output_modalities: ["text"],
		},
	},
	{
		id: "qwen3.6-35b-fast",
		object: "model",
		created: 0,
		owned_by: "neuralwatt",
		max_model_len: 262128,
		metadata: {
			display_name: "Qwen3.6 35B Fast",
			provider: "Qwen",
			huggingface_id: "Qwen/Qwen3.6-35B-A3B-FP8",
			pricing: {
				input_per_million: 0.29,
				output_per_million: 1.15,
				cached_input_per_million: 0.029,
				cached_output_per_million: null,
				currency: "USD",
				pricing_tbd: false,
				cached_input_is_default: true,
				service_tier: "standard",
				flex_discount_multiplier: null,
			},
			capabilities: {
				tools: true,
				json_mode: true,
				vision: true,
				audio_input: false,
				reasoning: false,
				reasoning_effort: true,
				streaming: true,
				system_role: true,
				developer_role: false,
				hosted_tools: false,
			},
			reasoning: {
				mandatory: false,
				default_enabled: false,
				supported_efforts: ["none"],
				default_effort: "none",
				accepted_efforts: ["max", "xhigh", "high", "medium", "low", "minimal", "none"],
				effort_aliases: {
					max: "none",
					xhigh: "none",
					high: "none",
					medium: "none",
					low: "none",
					minimal: "none",
				},
			},
			limits: {
				max_context_length: 262128,
				max_output_tokens: null,
				max_images: 20,
				max_images_total: 200,
				max_audio_seconds: null,
				max_audio_clips: null,
				max_audio_bytes: null,
			},
			deprecated: false,
			deprecated_message: null,
		},
		architecture: {
			modality: "text+image->text",
			input_modalities: ["text", "image"],
			output_modalities: ["text"],
		},
	},
	{
		id: "gemma-4-31b",
		object: "model",
		created: 0,
		owned_by: "neuralwatt",
		max_model_len: 262128,
		metadata: {
			display_name: "Gemma 4 31B",
			provider: "Google",
			huggingface_id: "nvidia/Gemma-4-31B-IT-NVFP4",
			pricing: {
				input_per_million: 0.144,
				output_per_million: 0.42,
				cached_input_per_million: 0.0144,
				cached_output_per_million: null,
				currency: "USD",
				pricing_tbd: false,
				cached_input_is_default: true,
				service_tier: "standard",
				flex_discount_multiplier: null,
			},
			capabilities: {
				tools: true,
				json_mode: true,
				vision: true,
				audio_input: false,
				reasoning: true,
				reasoning_effort: true,
				streaming: true,
				system_role: true,
				developer_role: false,
				hosted_tools: false,
			},
			reasoning: {
				mandatory: false,
				default_enabled: false,
				supported_efforts: ["max", "none"],
				default_effort: "none",
				accepted_efforts: ["max", "xhigh", "high", "medium", "low", "minimal", "none"],
				effort_aliases: {
					xhigh: "max",
					high: "max",
					medium: "max",
					low: "max",
					minimal: "max",
				},
			},
			limits: {
				max_context_length: 262128,
				max_output_tokens: 16384,
				max_images: 4,
				max_images_total: 4,
				max_audio_seconds: null,
				max_audio_clips: null,
				max_audio_bytes: null,
			},
			deprecated: false,
			deprecated_message: null,
		},
		architecture: {
			modality: "text+image->text",
			input_modalities: ["text", "image"],
			output_modalities: ["text"],
		},
	},
	{
		id: "nw-large",
		object: "model",
		created: 0,
		owned_by: "neuralwatt",
		max_model_len: 1048560,
		metadata: {
			display_name: "Neuralwatt Large",
			provider: "MoonshotAI",
			huggingface_id: "moonshotai/Kimi-K3.1-Pro",
			pricing: {
				input_per_million: 2.0,
				output_per_million: 10.0,
				cached_input_per_million: 0.2,
				cached_output_per_million: null,
				currency: "USD",
				pricing_tbd: false,
				cached_input_is_default: false,
				service_tier: "standard",
				flex_discount_multiplier: null,
			},
			capabilities: {
				tools: true,
				json_mode: true,
				vision: true,
				audio_input: false,
				reasoning: true,
				reasoning_effort: true,
				streaming: true,
				system_role: true,
				developer_role: false,
				hosted_tools: false,
			},
			reasoning: {
				mandatory: false,
				default_enabled: true,
				supported_efforts: ["max", "high", "medium", "low", "none"],
				default_effort: "max",
				accepted_efforts: ["max", "xhigh", "high", "medium", "low", "minimal", "none"],
				effort_aliases: {
					xhigh: "max",
					minimal: "low",
				},
			},
			limits: {
				max_context_length: 1048560,
				max_output_tokens: null,
				max_images: 20,
				max_images_total: 200,
				max_audio_seconds: null,
				max_audio_clips: null,
				max_audio_bytes: null,
			},
			deprecated: false,
			deprecated_message: null,
		},
		architecture: {
			modality: "text+image->text",
			input_modalities: ["text", "image"],
			output_modalities: ["text"],
		},
	},
	{
		id: "mammoth-1",
		object: "model",
		created: 0,
		owned_by: "neuralwatt",
		max_model_len: 32753,
		metadata: {
			display_name: "mammoth-1",
			provider: "Baseten",
			huggingface_id: "baselabs/mammoth-1",
			pricing: {
				input_per_million: 0.18,
				output_per_million: 0.0,
				cached_input_per_million: 0.018,
				cached_output_per_million: null,
				currency: "USD",
				pricing_tbd: false,
				cached_input_is_default: true,
				service_tier: "standard",
				flex_discount_multiplier: null,
			},
			capabilities: {
				tools: false,
				json_mode: false,
				vision: false,
				audio_input: false,
				reasoning: false,
				reasoning_effort: false,
				streaming: false,
				system_role: false,
				developer_role: false,
				task: "decision",
				hosted_tools: false,
			},
			reasoning: null,
			limits: {
				max_context_length: 32753,
				max_output_tokens: 0,
				max_images: null,
				max_images_total: null,
				max_audio_seconds: null,
				max_audio_clips: null,
				max_audio_bytes: null,
			},
			deprecated: false,
			deprecated_message: null,
		},
		architecture: {
			modality: "text->decisions",
			input_modalities: ["text"],
			output_modalities: ["decisions"],
		},
	},
];

function row(id: string, patch: (metadata: Record<string, unknown>) => void = () => {}): Record<string, unknown> {
	const source = LIVE_ROWS.find(entry => entry.id === id);
	if (!source) throw new Error(`fixture row ${id} missing`);
	const copy = structuredClone(source);
	patch(copy.metadata as Record<string, unknown>);
	return copy;
}

function catalogFetch(data: Record<string, unknown>[]): {
	authorizations: (string | null)[];
	urls: string[];
	fetch: FetchImpl;
} {
	const authorizations: (string | null)[] = [];
	const urls: string[] = [];
	const fetch: FetchImpl = async (input: string | URL | Request, init?: RequestInit) => {
		urls.push(String(input));
		authorizations.push(new Headers(init?.headers).get("authorization"));
		return Response.json({ object: "list", data });
	};
	return { authorizations, urls, fetch };
}

describe("Neuralwatt provider discovery", () => {
	test("discovers without a key and keeps only priced, live, tool-capable chat rows", async () => {
		const rows = [
			...LIVE_ROWS,
			row("nw-large", metadata => {
				metadata.deprecated = true;
			}),
			row("glm-5.3", metadata => {
				(metadata.pricing as Record<string, unknown>).pricing_tbd = true;
			}),
		];
		rows[rows.length - 2].id = "nw-retired";
		rows[rows.length - 1].id = "glm-unpriced";
		const { authorizations, urls, fetch } = catalogFetch(rows);
		const models = (await neuralwattModelManagerOptions({ fetch }).fetchDynamicModels?.()) ?? [];

		expect(urls).toEqual(["https://api.neuralwatt.com/v1/models"]);
		expect(authorizations).toEqual([null]);
		expect(models.map(model => model.id).sort()).toEqual(["gemma-4-31b", "glm-5.3", "nw-large", "qwen3.6-35b-fast"]);
		expect(models.find(model => model.id === "nw-large")).toMatchObject({
			provider: "neuralwatt",
			api: "openai-completions",
			baseUrl: "https://api.neuralwatt.com/v1",
			name: "Neuralwatt Large",
			input: ["text", "image"],
			supportsTools: true,
			cost: { input: 2, output: 10, cacheRead: 0.2, cacheWrite: 0 },
			contextWindow: 1048560,
			maxTokens: null,
		});
	});

	test("sends the key when one is configured", async () => {
		const { authorizations, fetch } = catalogFetch(LIVE_ROWS);
		await neuralwattModelManagerOptions({ apiKey: "sk-test", fetch }).fetchDynamicModels?.();
		expect(authorizations).toEqual(["Bearer sk-test"]);
	});

	test("builds effort ladders, output limits and the system role from live rows, fresh and from cache", async () => {
		const { fetch } = catalogFetch(LIVE_ROWS);
		const tempDir = await fs.mkdtemp(path.join(os.tmpdir(), "pi-catalog-neuralwatt-"));
		try {
			const options = { ...neuralwattModelManagerOptions({ fetch }), cacheDbPath: path.join(tempDir, "models.db") };
			for (const strategy of ["online", "offline"] as const) {
				const { models } = await resolveProviderModels(options, strategy);
				const byId = new Map(models.map(model => [model.id, model]));

				const large = byId.get("nw-large");
				expect(large?.thinking).toMatchObject({
					efforts: [Effort.Low, Effort.Medium, Effort.High, Effort.Max],
					defaultLevel: Effort.Max,
				});
				expect(large?.thinking?.requiresEffort).toBeUndefined();
				expect(large?.compat?.reasoningDisableMode).toBe("none-effort");
				// Null upstream output limit: the provider fallback seeds it.
				expect(large?.maxTokens).toBe(131072);
				expect(large?.compat?.supportsDeveloperRole).toBe(false);

				const glm = byId.get("glm-5.3");
				expect(glm?.thinking).toMatchObject({
					efforts: [Effort.Low, Effort.High, Effort.Max],
					requiresEffort: true,
				});
				expect(glm?.compat?.reasoningDisableMode).toBe("lowest-effort");

				// A reported output limit wins over the fallback.
				const gemma = byId.get("gemma-4-31b");
				expect(gemma?.maxTokens).toBe(16384);
				expect(gemma?.thinking?.efforts).toEqual([Effort.Max]);
				expect(gemma?.thinking?.defaultLevel).toBeUndefined();

				// No reasoning on the row: no effort dial is guessed from identity.
				const fast = byId.get("qwen3.6-35b-fast");
				expect(fast?.reasoning).toBe(false);
				expect(fast?.thinking).toBeUndefined();
			}
		} finally {
			await fs.rm(tempDir, { recursive: true, force: true });
		}
	});

	test("catalog generation keeps the live effort ladder", async () => {
		const { fetch } = catalogFetch(LIVE_ROWS);
		const specs = (await neuralwattModelManagerOptions({ fetch }).fetchDynamicModels?.()) ?? [];
		const models = specs.map(spec => buildModel(spec));
		applyGeneratedModelPolicies(models);
		const large = models.find(model => model.id === "nw-large");
		expect(large?.thinking?.efforts).toEqual([Effort.Low, Effort.Medium, Effort.High, Effort.Max]);
		expect(models.find(model => model.id === "qwen3.6-35b-fast")?.thinking).toBeUndefined();
	});
});
