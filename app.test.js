import assert from 'node:assert/strict';
import { request as sendRequest } from 'node:http';
import test from 'node:test';
import { createApp } from './app.js';

const createStore = () => {
  const values = new Map();
  return {
    async get(key) {
      return values.get(key);
    },
    async set(key, value) {
      values.set(key, value);
      return true;
    },
  };
};

const request = (server, path, options = {}) =>
  new Promise((resolve, reject) => {
    const address = server.address();
    const request = sendRequest({
      hostname: '127.0.0.1',
      port: address.port,
      path,
      method: options.method || 'GET',
      headers: options.headers,
    }, (response) => {
      let body = '';
      response.on('data', (chunk) => { body += chunk; });
      response.on('end', () => resolve({ response, body: body ? JSON.parse(body) : undefined }));
    });
    request.on('error', reject);
    if (options.body) request.write(options.body);
    request.end();
  });

test('creates a Redis-backed session and returns cached user data', async (t) => {
  const sessionStore = createStore();
  const userStore = createStore();
  let userLoads = 0;
  const server = createApp({
    sessionStore,
    userStore,
  userLoader: async (userId) => {
      userLoads += 1;
      return { id: userId, name: 'Ada' };
    },
    authenticateUser: async ({ email, password }) =>
      email === 'ada@example.com' && password === 'correct horse battery staple'
        ? { id: 42 }
        : null,
    sessionIdFactory: () => 'session-1',
  }).listen(0);
  t.after(() => server.close());

  const login = await request(server, '/session', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      email: 'ada@example.com',
      password: 'correct horse battery staple',
    }),
  });
  assert.equal(login.response.statusCode, 204);
  assert.equal(login.body, undefined);
  assert.match(login.response.headers['set-cookie'][0], /HttpOnly/);

  const cookie = login.response.headers['set-cookie'][0].split(';')[0];
  const firstUser = await request(server, '/user', {
    headers: { Cookie: cookie },
  });
  assert.deepEqual(firstUser.body, { id: 42, name: 'Ada' });

  const secondUser = await request(server, '/user', {
    headers: { Cookie: cookie },
  });
  assert.deepEqual(secondUser.body, firstUser.body);
  assert.equal(userLoads, 1);
});

test('does not create a session when authentication fails', async (t) => {
  const server = createApp({
    sessionStore: createStore(),
    userStore: createStore(),
    authenticateUser: async () => null,
  }).listen(0);
  t.after(() => server.close());

  const response = await request(server, '/session', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ email: 'ada@example.com', password: 'wrong' }),
  });

  assert.equal(response.response.statusCode, 401);
  assert.deepEqual(response.body, { error: 'Invalid credentials' });
  assert.equal(response.response.headers['set-cookie'], undefined);
});
