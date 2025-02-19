addEventListener('fetch', event => {
    event.respondWith(handleRequest(event.request));
});

async function handleRequest(request) {
    const TELEGRAM_BOT_TOKEN = globalThis.TELEGRAM_BOT_TOKEN; // Access the environment variable
    const telegramUrl = 'https://api.telegram.org/bot' + TELEGRAM_BOT_TOKEN + '/';
    const url = new URL(request.url);
    const path = url.pathname.split('/').pop();
    const newUrl = `${telegramUrl}${path}`;
    const init = {
        method: request.method,
        headers: request.headers,
        body: request.body
    };
    const response = await fetch(newUrl, init);
    return response;
}
