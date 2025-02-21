addEventListener('fetch', event => {
  event.respondWith(handleRequest(event.request));
});

async function handleRequest(request) {
  const TELEGRAM_BOT_TOKEN = globalThis.TELEGRAM_BOT_TOKEN;
  const telegramApiUrl = `https://api.telegram.org/bot${TELEGRAM_BOT_TOKEN}/sendMessage`;

  const url = new URL(request.url);
  const { pathname } = url;

  if (request.method === 'POST' && pathname === '/') {
    try {
      const update = await request.json();

      // Extract chat ID and message from the update
      const chatId = update.message.chat.id;
      const messageText = update.message.text;

      // Prepare your response message
      let responseMessage = '';

      if (messageText === '/start') {
        responseMessage = 'Welcome to the bot! Please enter the following properties of the material, separated by commas:\nDensity, H Cond, Sp Heat, Atm Mass, MFP Phonon, Atm R, Electrons';
      } else {
        // You can add your message handling logic here
        responseMessage = `You said: ${messageText}`;
      }

      // Send a message back to the user via Telegram API
      await fetch(telegramApiUrl, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          chat_id: chatId,
          text: responseMessage
        })
      });

      // Return a 200 OK response to Telegram
      return new Response('OK', { status: 200 });
    } catch (error) {
      console.error('Error handling update:', error);
      return new Response('Internal Server Error', { status: 500 });
    }
  } else {
    // Handle other requests or return 404
    return new Response('Not Found', { status: 404 });
  }
}
