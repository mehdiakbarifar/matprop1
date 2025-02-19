addEventListener('fetch', event => {
    event.respondWith(handleRequest(event.request))
})

async function handleRequest(request) {
    const telegramUrl = 'https://api.telegram.org/bot' + ENV.7877041621:AAGHM8hqQ55oNXkjoYyqm2Wz6VVciNqLm-Y + '/';
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
