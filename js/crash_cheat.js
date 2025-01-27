var OriginTitle = document.title;
var titleTime;
document.addEventListener('visibilitychange', function () {
    if (document.hidden) {
        $('[rel="icon"]').attr('href', "/joke.ico");
        document.title = '😠很不高興為你服務😠';
        clearTimeout(titleTime);
    }
    else {
        $('[rel="icon"]').attr('href', "/favicon.ico");
        document.title = '🥱很高興為你服務🥱';
        titleTime = setTimeout(function () {
            document.title = OriginTitle;
        }, 2000);
    }
});
