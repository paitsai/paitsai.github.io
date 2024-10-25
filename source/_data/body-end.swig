{# 鼠标点击特效 #}
{% if theme.cursor_effect == "fireworks" %}
  <script async src="/js/src/fireworks.js"></script>
{% elseif theme.cursor_effect == "explosion" %}
  <canvas class="fireworks" style="position: fixed;left: 0;top: 0;z-index: 1; pointer-events: none;" ></canvas>
  <script src="//cdn.bootcss.com/animejs/2.2.0/anime.min.js"></script>
  <script async src="/js/explosion.min.js"></script>
{% elseif theme.cursor_effect == "love" %}
  <script async src="/js/love.min.js"></script>
{% endif %}