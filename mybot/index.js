const restify = require('restify');
const { ActivityHandler, BotFrameworkAdapter } = require('botbuilder');

// Create adapter
const adapter = new BotFrameworkAdapter({
    appId: process.env.MICROSOFT_APP_ID || '',
    appPassword: process.env.MICROSOFT_APP_PASSWORD || ''
});

// Create bot logic
class EchoBot extends ActivityHandler {
    constructor() {
        super();
        this.onMessage(async (context, next) => {
            const userMessage = context.activity.text;
            await context.sendActivity(`You said: ${userMessage}`);
            await next();
        });
    }
}

const bot = new EchoBot();

// Create server
let server = restify.createServer();
server.listen(3978, () => {
    console.log(`\nBot listening on http://localhost:3978`);
});

server.post('/api/messages', async (req, res) => {
    await adapter.processActivity(req, res, async (context) => {
        await bot.run(context);
    });
});