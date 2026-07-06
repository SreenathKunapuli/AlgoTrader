import { defineConfig } from "@playwright/test";

export default defineConfig({
  testDir: "./e2e",
  timeout: 30_000,
  use: { baseURL: "http://localhost:3100" },
  webServer: {
    command: "npm run build >/dev/null 2>&1 || true; npx next start -p 3100",
    port: 3100,
    reuseExistingServer: true,
    timeout: 120_000,
  },
});
