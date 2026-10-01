import assert from 'node:assert/strict';
import test from 'node:test';
import { createCacheAside, DEFAULT_CACHE_TTL } from './cache.js';

const createMemoryStore = (now = () => 0) => {
  const entries = new Map();

  return {
    async get(key) {
      const entry = entries.get(key);
      if (!entry || entry.expiresAt <= now()) {
        return undefined;
      }
      return entry.value;
    },
    async set(key, value, ttl) {
      entries.set(key, {
        value,
        expiresAt: ttl === undefined ? Infinity : now() + ttl,
        ttl,
      });
      return true;
    },
    getTtl(key) {
      return entries.get(key)?.ttl;
    },
  };
};

test('returns cached values without calling the loader', async () => {
  const cacheStore = createMemoryStore();
  await cacheStore.set('profile:1', { name: 'Ada' }, DEFAULT_CACHE_TTL);
  const getOrLoad = createCacheAside(cacheStore);

  const value = await getOrLoad('profile:1', async () => {
    assert.fail('loader should not run on a cache hit');
  });

  assert.deepEqual(value, { name: 'Ada' });
});

test('loads and caches values on a miss using the default TTL', async () => {
  const cacheStore = createMemoryStore();
  const getOrLoad = createCacheAside(cacheStore);
  const value = { name: 'Grace' };

  assert.deepEqual(await getOrLoad('profile:2', async () => value), value);
  assert.deepEqual(await cacheStore.get('profile:2'), value);
  assert.equal(cacheStore.getTtl('profile:2'), DEFAULT_CACHE_TTL);
});

test('reloads values after their TTL expires', async () => {
  let now = 100;
  const cacheStore = createMemoryStore(() => now);
  const getOrLoad = createCacheAside(cacheStore);

  await getOrLoad('feed', async () => 'old', 20);
  now += 20;
  const value = await getOrLoad('feed', async () => 'fresh', 20);

  assert.equal(value, 'fresh');
});

test('coalesces concurrent cache misses for the same key', async () => {
  const cacheStore = createMemoryStore();
  const getOrLoad = createCacheAside(cacheStore);
  let loaderCalls = 0;
  let releaseLoader;
  const loaderStarted = new Promise((resolve) => {
    releaseLoader = resolve;
  });

  const loader = async () => {
    loaderCalls += 1;
    await loaderStarted;
    return { name: 'Ada' };
  };

  const first = getOrLoad('profile:5', loader);
  const second = getOrLoad('profile:5', loader);
  await new Promise((resolve) => setImmediate(resolve));
  assert.equal(loaderCalls, 1);

  releaseLoader();
  assert.deepEqual(await first, { name: 'Ada' });
  assert.deepEqual(await second, { name: 'Ada' });
});

test('returns loaded values when cache reads and writes fail', async () => {
  const cacheStore = {
    async get() {
      throw new Error('read failed');
    },
    async set() {
      throw new Error('write failed');
    },
  };
  const getOrLoad = createCacheAside(cacheStore, DEFAULT_CACHE_TTL, () => {});
  let loaderCalls = 0;

  const value = await getOrLoad('profile:3', async () => {
    loaderCalls += 1;
    return { name: 'Lin' };
  });

  assert.equal(loaderCalls, 1);
  assert.deepEqual(value, { name: 'Lin' });
});

test('does not cache a value when the loader fails', async () => {
  const cacheStore = createMemoryStore();
  const getOrLoad = createCacheAside(cacheStore);
  const error = new Error('loader failed');

  await assert.rejects(
    getOrLoad('profile:4', async () => {
      throw error;
    }),
    error,
  );

  assert.equal(await cacheStore.get('profile:4'), undefined);
});
