(function () {
  "use strict";

  var counter = document.querySelector("[data-view-counter]");
  if (!counter) return;

  var endpoint = (counter.dataset.endpoint || "").replace(/\/+$/, "");
  var articleId = counter.dataset.articleId;
  var countElement = counter.querySelector("[data-view-count]");
  var article = document.querySelector("article.page.single:not(.special)");
  if (!endpoint || !articleId || !countElement || !article) return;

  function formatCount(value) {
    try {
      return new Intl.NumberFormat(document.documentElement.lang || undefined).format(value);
    } catch (error) {
      return String(value);
    }
  }

  counter.hidden = false;
  counter.dataset.state = "loading";
  counter.setAttribute("aria-busy", "true");

  fetch(endpoint + "/api/v1/articles/" + encodeURIComponent(articleId) + "/reads", {
    method: "POST",
    mode: "cors",
    credentials: "omit",
    keepalive: true,
  })
    .then(function (response) {
      if (!response.ok) throw new Error("view counter request failed");
      return response.json();
    })
    .then(function (payload) {
      var views = payload && payload.data && Number(payload.data.total_views);
      if (!Number.isFinite(views)) throw new Error("view counter response invalid");
      countElement.textContent = formatCount(views);
      counter.dataset.state = "ready";
      counter.setAttribute("aria-busy", "false");
    })
    .catch(function () {
      counter.hidden = true;
      counter.dataset.state = "unavailable";
      counter.removeAttribute("aria-busy");
    });
})();
