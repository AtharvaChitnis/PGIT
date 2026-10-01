import 'dotenv/config';
import express from 'express';
import cookieParser from 'cookie-parser';
import { randomUUID } from 'node:crypto';
import { pathToFileURL } from 'node:url';
import { createCacheAside, createRedisStore } from './cache.js';

const SESSION_COOKIE = 'sessionId';
const SESSION_TTL = 7 * 24 * 60 * 60 * 1000;
const USER_CACHE_TTL = 5 * 60 * 1000;
const MAX_SESSION_ID_LENGTH = 128;

const defaultUserLoader = async (userId) => ({
  id: Number(userId),
  name: 'Atharva',
  email: 'user@example.com',
});

const defaultAuthenticator = async () => null;

const isPositiveInteger = (value) => Number.isSafeInteger(value) && value > 0;

const isSessionId = (value) =>
  typeof value === 'string' && value.length > 0 && value.length <= MAX_SESSION_ID_LENGTH;

const cookieOptions = () => [
  'Path=/',
  'HttpOnly',
  'SameSite=Lax',
  ...(process.env.NODE_ENV === 'production' ? ['Secure'] : []),
  `Max-Age=${SESSION_TTL / 1000}`,
];

export const createApp = ({
  sessionStore = createRedisStore('sessions'),
  userStore = createRedisStore('users'),
  userLoader = defaultUserLoader,
  authenticateUser = defaultAuthenticator,
  sessionIdFactory = randomUUID,
} = {}) => {
  const app = express();
  const getOrLoadUser = createCacheAside(userStore, USER_CACHE_TTL);

  app.use(express.json());
  app.use(cookieParser());

  app.get('/healthz', (_request, response) => {
    response.json({ status: 'ok' });
  });

  app.post('/session', async (request, response, next) => {
    try {
      const user = await authenticateUser(request.body, request);
      const userId = user?.id;
      if (!isPositiveInteger(userId)) {
        return response.status(401).json({ error: 'Invalid credentials' });
      }

      const sessionId = sessionIdFactory();
      await sessionStore.set(`session:${sessionId}`, { userId }, SESSION_TTL);
      response.setHeader('Set-Cookie', `${SESSION_COOKIE}=${encodeURIComponent(sessionId)}; ${cookieOptions().join('; ')}`);
      return response.status(204).end();
    } catch (error) {
      return next(error);
    }
  });

  app.get('/user', async (request, response, next) => {
    try {
      const sessionId = request.cookies[SESSION_COOKIE];
      if (!isSessionId(sessionId)) {
        return response.status(401).json({ error: 'Session cookie is required' });
      }

      const session = await sessionStore.get(`session:${sessionId}`);
      if (!isPositiveInteger(session?.userId)) {
        return response.status(401).json({ error: 'Session is invalid or expired' });
      }

      const user = await getOrLoadUser(`user:${session.userId}`, () => userLoader(session.userId));
      return response.json(user);
    } catch (error) {
      return next(error);
    }
  });

  app.use((error, _request, response, _next) => {
    console.error('[API Error]', error);
    response.status(500).json({ error: 'Internal server error' });
  });

  return app;
};

const isMainModule = process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href;
export const app = isMainModule ? createApp() : undefined;

if (isMainModule) {
  const port = Number(process.env.PORT) || 3000;
  app.listen(port, () => {
    console.log(`API listening on port ${port}`);
  });
}

export { SESSION_TTL, USER_CACHE_TTL };
