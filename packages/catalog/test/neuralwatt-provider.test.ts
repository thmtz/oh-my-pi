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

/** A `/v1/models` row (https://docs.neuralwatt.com/api/models) shaped like live `nw-large`. */
function nwRow(id: string, metadata: Record<string, unknown> = {}): Record<string, unknown> {
	return {
		id,
		object: "model",
		max_model_len: 1048560,
		metadata: {
			display_name: id,
			pricing: { input_per_million: 2, output_per_million: 10, cached_input_per_million: 0.2, pricing_tbd: false },
			capabilities: { tools: true, vision: true, reasoning: true, developer_role: false },
			reasoning: {
				mandatory: false,
				supported_efforts: ["max", "high", "medium", "low", "none"],
				default_effort: "max",
				effort_aliases: { xhigh: "max", minimal: "low" },
			},
			limits: { max_context_length: 1048560, max_output_tokens: null },
			deprecated: false,
			...metadata,
		},
	};
}

const ROWS = [
	nwRow("nw-large"),
	nwRow("kimi-k3.1-pro"),
	nwRow("glm-unpriced", { pricing: { input_per_million: 1, output_per_million: 3, pricing_tbd: true } }),
	nwRow("qwen-no-reasoning", { capabilities: { tools: true, reasoning: false } }),
	nwRow("nw-retired", { deprecated: true }),
	// Decision models such as `mammoth-1` cannot call tools.
	nwRow("mammoth-1", { capabilities: { tools: false, reasoning: false }, reasoning: null }),
];

/** Serves `ROWS`, plus a private row to `sk-account-a`. Not a real model id. */
function catalogFetch(): { authorizations: (string | null)[]; fetch: FetchImpl } {
	const authorizations: (string | null)[] = [];
	const fetch: FetchImpl = async (_input: string | URL | Request, init?: RequestInit) => {
		const authorization = new Headers(init?.headers).get("authorization");
		authorizations.push(authorization);
		const data = authorization === "Bearer sk-account-a" ? [...ROWS, nwRow("nw-test-private")] : ROWS;
		return Response.json({ object: "list", data });
	};
	return { authorizations, fetch };
}

describe("Neuralwatt provider discovery", () => {
	test("keeps live, tool-capable rows and marks unpriced rows unknown instead of free", async () => {
		const models = (await neuralwattModelManagerOptions(catalogFetch()).fetchDynamicModels?.()) ?? [];
		expect(models.map(model => model.id).sort()).toEqual([
			"glm-unpriced",
			"kimi-k3.1-pro",
			"nw-large",
			"qwen-no-reasoning",
		]);
		expect(models.find(model => model.id === "glm-unpriced")).toMatchObject({
			pricingStatus: "unknown",
			cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
		});
		expect(models.find(model => model.id === "nw-large")).toMatchObject({
			baseUrl: "https://api.neuralwatt.com/v1",
			input: ["text", "image"],
			cost: { input: 2, output: 10, cacheRead: 0.2, cacheWrite: 0 },
			contextWindow: 1048560,
			maxTokens: null,
		});
	});

	test("keeps each key's private rows out of other keys' and keyless caches", async () => {
		await using temp = await TempDir.create("@neuralwatt-credential-cache-");
		const { authorizations, fetch } = catalogFetch();
		const ids = async (apiKey: string | undefined, strategy: "online-if-uncached" | "offline") => {
			const options = { ...neuralwattModelManagerOptions({ apiKey, fetch }), cacheDbPath: temp.join("models.db") };
			return (await resolveProviderModels(options, strategy)).models.map(model => model.id);
		};
		for (const strategy of ["online-if-uncached", "offline"] as const) {
			expect(await ids("sk-account-a", strategy)).toContain("nw-test-private");
			expect(await ids("sk-account-b", strategy)).not.toContain("nw-test-private");
			expect(await ids(undefined, strategy)).not.toContain("nw-test-private");
		}
		// The offline pass reads each slot's cache without fetching again.
		expect(authorizations).toEqual(["Bearer sk-account-a", "Bearer sk-account-b", null]);
	});

	test("catalog generation sends no key and keeps the live effort ladder", async () => {
		const descriptor = (id: string) => {
			const found = PROVIDER_DESCRIPTORS.find(entry => entry.providerId === id);
			if (!found || !isCatalogDescriptor(found)) throw new Error(`catalog descriptor ${id} missing`);
			return found;
		};
		const previous = { NEURALWATT_API_KEY: Bun.env.NEURALWATT_API_KEY, CHUTES_API_KEY: Bun.env.CHUTES_API_KEY };
		Bun.env.NEURALWATT_API_KEY = "sk-account-a";
		Bun.env.CHUTES_API_KEY = "cpk-generation";
		try {
			expect(await resolveCatalogDiscoveryApiKey(descriptor("neuralwatt"))).toBeUndefined();
			// Other providers still read the environment.
			expect(await resolveCatalogDiscoveryApiKey(descriptor("chutes"))).toBe("cpk-generation");
		} finally {
			for (const [name, value] of Object.entries(previous)) {
				if (value === undefined) delete Bun.env[name];
				else Bun.env[name] = value;
			}
		}

		const specs = (await neuralwattModelManagerOptions(catalogFetch()).fetchDynamicModels?.()) ?? [];
		const models = specs.map(spec => buildModel(spec));
		applyGeneratedModelPolicies(models);
		const kimi = models.find(model => model.id === "kimi-k3.1-pro");
		expect(kimi?.thinking?.efforts).toEqual([Effort.Low, Effort.Medium, Effort.High, Effort.Max]);
		// No reasoning on the row, so no effort selector guessed from model identity.
		expect(models.find(model => model.id === "qwen-no-reasoning")?.thinking).toBeUndefined();
	});
});
