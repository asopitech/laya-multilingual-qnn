const baseUrl = process.env.LOCAL_JEV_URL ?? "http://127.0.0.1:8788";
const response = await fetch(`${baseUrl}/decision`, {
  method: "POST",
  headers: { "content-type": "application/json" },
  body: JSON.stringify({
    state: {
      subject: "決済障害",
      body: "購入者全員が支払いできません。至急、技術担当に調査してほしいです。",
    },
    questions: {
      team: {
        type: "choice",
        instructions: "どの担当チームが対応すべきですか？",
        criteria: {
          billing: "請求、返金、重複課金",
          engineering: "障害、バグ、機能停止",
          sales: "新規契約、価格、購入相談",
        },
      },
      outage: {
        type: "noul",
        instructions: "サービス障害が発生していますか？",
      },
    },
  }),
});

if (!response.ok) {
  throw new Error(`${response.status}: ${await response.text()}`);
}

console.log(JSON.stringify(await response.json(), null, 2));

