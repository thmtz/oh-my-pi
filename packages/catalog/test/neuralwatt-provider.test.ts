import { describe, expect, test } from "bun:test";
import { buildModel } from "@oh-my-pi/pi-catalog/build";
import { Effort } from "@oh-my-pi/pi-catalog/effort";
import { resolveProviderModels } from "@oh-my-pi/pi-catalog/model-manager";
import { isCatalogDescriptor } from "@oh-my-pi/pi-catalog/provider-models/descriptor-types";
import { PROVIDER_DESCRIPTORS } from "@oh-my-pi/pi-catalog/provider-models/descriptors";
import { neuralwattModelManagerOptions } from "@oh-my-pi/pi-catalog/provider-models/openai-compat";
import type { FetchImpl } from "@oh-my-pi/pi-catalog/types";
import { TempDir } from "@oh-my-pi/pi-utils";
import { resolveCatalogDiscoveryApiKey } from "../scripts/generate-models";
import { applyGeneratedModelPolicies } from "../scripts/generated-policies";

/**
 * Rows copied from the public `https://api.neuralwatt.com/v1/models`
 * (2026-11-20), descriptions trimmed. They cover the shapes the mapper
 * branches on:
 * - `nw-large`: tracking alias, null `max_output_tokens`, optional reasoning
 *   with a `none` off value
 * - `kimi-k3.1-pro`: the same row shape under an id that Kimi K3 class rules
 *   recognize
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
		id: "kimi-k3.1-pro",
		object: "model",
		created: 0,
		owned_by: "neuralwatt",
		max_model_len: 1048560,
		metadata: {
			display_name: "Kimi K3.1 Pro",
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

function row(id: string, patch: (copy: Record<string, unknown>) => void = () => {}): Record<string, unknown> {
	const source = LIVE_ROWS.find(entry => entry.id === id);
	if (!source) throw new Error(`fixture row ${id} missing`);
	const copy = structuredClone(source);
	patch(copy);
	return copy;
}

function metadataOf(entry: Record<string, unknown>): Record<string, unknown> {
	return entry.metadata as Record<string, unknown>;
}

/** A row only an authenticated request returns. Not a real model id. */
const PRIVATE_ROW = row("nw-large", entry => {
	entry.id = "nw-test-private";
});

/** Serves the public roster, plus `PRIVATE_ROW` to `sk-account-a`. */
function catalogFetch(): { authorizations: (string | null)[]; urls: string[]; fetch: FetchImpl } {
	const authorizations: (string | null)[] = [];
	const urls: string[] = [];
	const fetch: FetchImpl = async (input: string | URL | Request, init?: RequestInit) => {
		const authorization = new Headers(init?.headers).get("authorization");
		urls.push(String(input));
		authorizations.push(authorization);
		const data = authorization === "Bearer sk-account-a" ? [...LIVE_ROWS, PRIVATE_ROW] : LIVE_ROWS;
		return Response.json({ object: "list", data });
	};
	return { authorizations, urls, fetch };
}

describe("Neuralwatt provider discovery", () => {
	test("keeps live, tool-capable chat rows and marks unpriced rows unknown instead of free", async () => {
		const deprecated = row("nw-large", entry => {
			entry.id = "nw-retired";
			metadataOf(entry).deprecated = true;
		});
		const unpriced = row("glm-5.3", entry => {
			entry.id = "glm-unpriced";
			(metadataOf(entry).pricing as Record<string, unknown>).pricing_tbd = true;
		});
		const fetch: FetchImpl = async () => Response.json({ data: [...LIVE_ROWS, deprecated, unpriced] });
		const models = (await neuralwattModelManagerOptions({ fetch }).fetchDynamicModels?.()) ?? [];

		expect(models.map(model => model.id).sort()).toEqual([
			"gemma-4-31b",
			"glm-5.3",
			"glm-unpriced",
			"kimi-k3.1-pro",
			"nw-large",
			"qwen3.6-35b-fast",
		]);
		expect(models.find(model => model.id === "glm-unpriced")).toMatchObject({
			pricingStatus: "unknown",
			cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
		});
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

	test("keeps each key's private rows out of other keys' and keyless caches", async () => {
		await using temp = await TempDir.create("@neuralwatt-credential-cache-");
		const { authorizations, fetch } = catalogFetch();
		const options = (apiKey?: string) => ({
			...neuralwattModelManagerOptions({ apiKey, fetch }),
			cacheDbPath: temp.join("models.db"),
		});
		const ids = async (apiKey: string | undefined, strategy: "online-if-uncached" | "offline") =>
			(await resolveProviderModels(options(apiKey), strategy)).models.map(model => model.id);

		expect(await ids("sk-account-a", "online-if-uncached")).toContain("nw-test-private");
		expect(await ids("sk-account-b", "online-if-uncached")).not.toContain("nw-test-private");
		expect(await ids(undefined, "online-if-uncached")).not.toContain("nw-test-private");
		expect(authorizations).toEqual(["Bearer sk-account-a", "Bearer sk-account-b", null]);

		// Each slot serves its own cached roster without fetching again.
		expect(await ids("sk-account-a", "offline")).toContain("nw-test-private");
		expect(await ids("sk-account-b", "offline")).not.toContain("nw-test-private");
		expect(await ids(undefined, "offline")).not.toContain("nw-test-private");
		expect(authorizations).toHaveLength(3);
	});

	test("builds ladders, output limits and the system role from live rows, fresh and from cache", async () => {
		await using temp = await TempDir.create("@neuralwatt-policy-cache-");
		const { urls, fetch } = catalogFetch();
		const options = {
			...neuralwattModelManagerOptions({ apiKey: "sk-account-a", fetch }),
			cacheDbPath: temp.join("models.db"),
		};
		for (const strategy of ["online-if-uncached", "offline"] as const) {
			const { models } = await resolveProviderModels(options, strategy);
			const byId = new Map(models.map(model => [model.id, model]));
			// The private row is never bundled, so the offline pass reads it from cache.
			expect(byId.has("nw-test-private")).toBe(true);

			// The alias and the id the Kimi K3 class rules recognize get the
			// same row policy: optional reasoning, native `medium`, `none` for off.
			for (const id of ["nw-large", "kimi-k3.1-pro"]) {
				const model = byId.get(id);
				expect(model?.thinking).toMatchObject({
					efforts: [Effort.Low, Effort.Medium, Effort.High, Effort.Max],
					defaultLevel: Effort.Max,
					requiresEffort: false,
				});
				// Served efforts map to themselves, and the row's aliases fill the rest.
				const effortMap = { minimal: "low", low: "low", medium: "medium", high: "high", xhigh: "max", max: "max" };
				expect(model?.thinking?.effortMap).toEqual(effortMap);
				expect(model?.compat?.reasoningEffortMap).toEqual(effortMap);
				expect(model?.compat?.reasoningDisableMode).toBe("none-effort");
				// The provider fallback seeds the null upstream output limit.
				expect(model?.maxTokens).toBe(131072);
				expect(model?.compat?.supportsDeveloperRole).toBe(false);
			}

			const glm = byId.get("glm-5.3");
			expect(glm?.thinking).toMatchObject({ efforts: [Effort.Low, Effort.High, Effort.Max], requiresEffort: true });
			expect(glm?.thinking?.effortMap?.[Effort.Medium]).toBe("high");
			expect(glm?.compat?.reasoningDisableMode).toBe("lowest-effort");

			// A reported output limit wins over the fallback.
			const gemma = byId.get("gemma-4-31b");
			expect(gemma?.maxTokens).toBe(16384);
			expect(gemma?.thinking?.efforts).toEqual([Effort.Max]);
			expect(gemma?.thinking?.defaultLevel).toBeUndefined();

			// No reasoning on the row, so no effort selector from model identity.
			const fast = byId.get("qwen3.6-35b-fast");
			expect(fast?.reasoning).toBe(false);
			expect(fast?.thinking).toBeUndefined();
		}
		expect(urls).toEqual(["https://api.neuralwatt.com/v1/models"]);
	});

	test("catalog generation keeps the live effort ladder", async () => {
		const { fetch } = catalogFetch();
		const specs = (await neuralwattModelManagerOptions({ fetch }).fetchDynamicModels?.()) ?? [];
		const models = specs.map(spec => buildModel(spec));
		applyGeneratedModelPolicies(models);
		const kimi = models.find(model => model.id === "kimi-k3.1-pro");
		expect(kimi?.thinking?.efforts).toEqual([Effort.Low, Effort.Medium, Effort.High, Effort.Max]);
		expect(kimi?.thinking?.requiresEffort).toBe(false);
		expect(kimi?.thinking?.effortMap).toMatchObject({ medium: "medium", xhigh: "max", minimal: "low" });
		expect(models.find(model => model.id === "qwen3.6-35b-fast")?.thinking).toBeUndefined();
	});

	test("catalog generation never sends a key, so private rows stay out of the bundle", async () => {
		const descriptor = (id: string) => {
			const found = PROVIDER_DESCRIPTORS.find(entry => entry.providerId === id);
			if (!found || !isCatalogDescriptor(found)) throw new Error(`catalog descriptor ${id} missing`);
			return found;
		};
		const previous = { neuralwatt: Bun.env.NEURALWATT_API_KEY, chutes: Bun.env.CHUTES_API_KEY };
		Bun.env.NEURALWATT_API_KEY = "sk-account-a";
		Bun.env.CHUTES_API_KEY = "cpk-generation";
		try {
			expect(await resolveCatalogDiscoveryApiKey(descriptor("neuralwatt"))).toBeUndefined();
			// Other providers still read the environment.
			expect(await resolveCatalogDiscoveryApiKey(descriptor("chutes"))).toBe("cpk-generation");
		} finally {
			for (const [name, value] of [
				["NEURALWATT_API_KEY", previous.neuralwatt],
				["CHUTES_API_KEY", previous.chutes],
			] as const) {
				if (value === undefined) delete Bun.env[name];
				else Bun.env[name] = value;
			}
		}
	});
});
