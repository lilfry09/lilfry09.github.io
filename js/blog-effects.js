(function () {
  "use strict";

  const motionQuery = window.matchMedia("(prefers-reduced-motion: reduce)");
  const clickText = "💜感谢观看💜";
  const leaveTitle = "(つд⊂) 不要走呀";
  const returnTitle = "(*´▽`*) 欢迎回来!";
  const clickPalettes = [
    { hue: 269, saturation: 92, lightness: 62, color: "#9a45ff", glow: "rgba(154, 69, 255, 0.24)" },
    { hue: 31, saturation: 94, lightness: 58, color: "#f28c28", glow: "rgba(242, 140, 40, 0.24)" },
    { hue: 338, saturation: 88, lightness: 66, color: "#f45f9a", glow: "rgba(244, 95, 154, 0.24)" },
    { hue: 145, saturation: 58, lightness: 48, color: "#35b86f", glow: "rgba(53, 184, 111, 0.22)" },
    { hue: 205, saturation: 78, lightness: 54, color: "#2f9fe8", glow: "rgba(47, 159, 232, 0.24)" },
    { hue: 25, saturation: 48, lightness: 42, color: "#9a6840", glow: "rgba(154, 104, 64, 0.24)" }
  ];
  const originalTitle = document.title;
  let restoreTitleTimer = 0;
  let toastTimer = 0;
  let clickPaletteIndex = 0;

  function ensureLayer() {
    let layer = document.querySelector(".blog-click-effects");
    if (!layer) {
      layer = document.createElement("div");
      layer.className = "blog-click-effects";
      layer.setAttribute("aria-hidden", "true");
      document.body.appendChild(layer);
    }
    return layer;
  }

  function randomBetween(min, max) {
    return Math.random() * (max - min) + min;
  }

  function getNextClickPalette() {
    const palette = clickPalettes[clickPaletteIndex % clickPalettes.length];
    clickPaletteIndex += 1;
    return palette;
  }

  function createParticle(layer, x, y, index, palette) {
    const particle = document.createElement("span");
    const angle = randomBetween(0, Math.PI * 2);
    const distance = randomBetween(52, 138);
    const size = randomBetween(8, 20);
    const duration = randomBetween(720, 1180);
    const delay = index * 8;

    particle.className = "blog-click-particle";
    particle.style.left = `${x}px`;
    particle.style.top = `${y}px`;
    particle.style.width = `${size}px`;
    particle.style.height = `${size}px`;
    particle.style.setProperty("--tx", `${Math.cos(angle) * distance}px`);
    particle.style.setProperty("--ty", `${Math.sin(angle) * distance}px`);
    particle.style.setProperty("--duration", `${duration}ms`);
    particle.style.setProperty("--delay", `${delay}ms`);
    particle.style.setProperty("--hue", String(Math.round(randomBetween(palette.hue - 8, palette.hue + 8))));
    particle.style.setProperty("--saturation", `${palette.saturation}%`);
    particle.style.setProperty("--lightness", `${randomBetween(palette.lightness - 5, palette.lightness + 6).toFixed(1)}%`);
    layer.appendChild(particle);

    window.setTimeout(function () {
      particle.remove();
    }, duration + delay + 80);
  }

  function createRing(layer, x, y, palette) {
    const ring = document.createElement("span");
    ring.className = "blog-click-ring";
    ring.style.left = `${x}px`;
    ring.style.top = `${y}px`;
    ring.style.setProperty("--effect-color", palette.color);
    ring.style.setProperty("--effect-glow", palette.glow);
    layer.appendChild(ring);

    window.setTimeout(function () {
      ring.remove();
    }, 920);
  }

  function createClickText(layer, x, y, palette) {
    const label = document.createElement("span");
    label.className = "blog-click-text";
    label.textContent = clickText;
    label.style.left = `${x}px`;
    label.style.top = `${y}px`;
    label.style.setProperty("--effect-color", palette.color);
    label.style.setProperty("--effect-glow", palette.glow);
    layer.appendChild(label);

    window.setTimeout(function () {
      label.remove();
    }, 1200);
  }

  function showClickEffect(event) {
    if (motionQuery.matches || event.button !== 0) return;
    if (event.target && event.target.closest(".blog-visibility-toast, .blog-click-effects")) return;

    const layer = ensureLayer();
    const x = event.clientX;
    const y = event.clientY;
    const palette = getNextClickPalette();

    createRing(layer, x, y, palette);
    createClickText(layer, x, y, palette);

    for (let i = 0; i < 34; i += 1) {
      createParticle(layer, x, y, i, palette);
    }
  }

  function showWelcomeToast() {
    let toast = document.querySelector(".blog-visibility-toast");
    if (!toast) {
      toast = document.createElement("div");
      toast.className = "blog-visibility-toast";
      toast.setAttribute("role", "status");
      toast.setAttribute("aria-live", "polite");
      toast.innerHTML = '<span class="blog-visibility-toast-icon">↻</span><span class="blog-visibility-toast-text"></span><button class="blog-visibility-toast-close" type="button" aria-label="关闭提示">×</button>';
      document.body.appendChild(toast);

      toast.querySelector(".blog-visibility-toast-close").addEventListener("click", function () {
        toast.classList.remove("is-visible");
      });
    }

    toast.querySelector(".blog-visibility-toast-text").textContent = returnTitle;
    window.clearTimeout(toastTimer);
    requestAnimationFrame(function () {
      toast.classList.add("is-visible");
    });
    toastTimer = window.setTimeout(function () {
      toast.classList.remove("is-visible");
    }, 2600);
  }

  document.addEventListener("click", showClickEffect, { passive: true });

  document.addEventListener("visibilitychange", function () {
    window.clearTimeout(restoreTitleTimer);

    if (document.hidden) {
      document.title = leaveTitle;
      restoreTitleTimer = window.setTimeout(function () {
        if (document.hidden) {
          document.title = originalTitle;
        }
      }, 2500);
      return;
    }

    document.title = returnTitle;
    showWelcomeToast();
    restoreTitleTimer = window.setTimeout(function () {
      document.title = originalTitle;
    }, 2500);
  });
})();
