// Smoke test (§7): login -> dashboard renders -> tier dialog opens ->
// kill dialog requires the typed FLATTEN confirmation.
// The backend is mocked at the network layer so the test runs standalone.
import { expect, Page, test } from "@playwright/test";

async function mockApi(page: Page) {
  await page.route("**/api/auth/login", (route) =>
    route.fulfill({
      status: 200,
      headers: {
        "content-type": "application/json",
        "set-cookie": "lob_jwt=test-token; Path=/; HttpOnly; SameSite=Lax",
      },
      body: JSON.stringify({ ok: true }),
    })
  );
  const canned: Record<string, unknown> = {
    "engine/status": {
      status: "RUNNING", tier: "medium", halted_reason: "",
      heartbeat_age_s: 3, last_data_ts: null, trading_mode: "paper",
    },
    account: {
      equity: 100_000, cash: 60_000, gross_exposure: 40_000,
      day_pnl: 250.5, day_pnl_pct: 0.25, buying_power: 60_000,
    },
    positions: [
      { symbol: "SPY", qty: 10, entry: 500.0, mark: 501.2, upnl: 12.0, stop: 495.5 },
    ],
    "equity-curve": [{ ts: new Date().toISOString(), equity: 100_000 }],
  };
  await page.route("**/api/proxy/**", (route) => {
    const url = new URL(route.request().url());
    const path = url.pathname.replace(/^\/api\/proxy\//, "").replace(/\/$/, "");
    const body = canned[path] ?? [];
    route.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify(body) });
  });
  await page.route("**/api/ws-token", (route) =>
    route.fulfill({ status: 401, contentType: "application/json", body: "{}" })
  );
}

test("login -> dashboard -> tier dialog -> kill dialog typed confirm", async ({ page, context }) => {
  await mockApi(page);

  // unauthenticated hit redirects to /login
  await page.goto("/");
  await expect(page).toHaveURL(/\/login/);

  // login (route.fulfill set-cookie doesn't hit the jar; seed it explicitly —
  // the real cookie path is covered by the Next route handler + API tests)
  await context.addCookies([
    { name: "lob_jwt", value: "test-token", url: "http://localhost:3100" },
  ]);
  await page.getByTestId("password-input").fill("pw");
  await page.getByTestId("login-button").click();
  await page.goto("/");  // hard navigation so middleware sees the cookie jar
  await expect(page).toHaveURL(/\/$/);

  // dashboard renders: status pill + equity card + position row
  await expect(page.getByTestId("status-pill")).toHaveText("RUNNING");
  await expect(page.getByText("$100,000")).toBeVisible();
  await expect(page.getByText("SPY")).toBeVisible();

  // tier dialog opens with limits text
  await page.getByTestId("tier-selector").getByText("high").click();
  await expect(page.getByTestId("tier-dialog")).toBeVisible();
  await expect(page.getByTestId("tier-dialog")).toContainText("20% max position");
  await page.getByText("Cancel").click();

  // kill dialog requires typing FLATTEN
  await page.getByTestId("kill-button").click();
  await expect(page.getByTestId("kill-dialog")).toBeVisible();
  const confirmBtn = page.getByTestId("kill-confirm-button");
  await expect(confirmBtn).toBeDisabled();
  await page.getByTestId("kill-confirm-input").fill("WRONG");
  await expect(confirmBtn).toBeDisabled();
  await page.getByTestId("kill-confirm-input").fill("FLATTEN");
  await expect(confirmBtn).toBeEnabled();
});
