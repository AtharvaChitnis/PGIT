import Keyv from 'keyv';
import KeyvRedis from '@keyv/redis';

const DEFAULT_REDIS_URL = 'redis://default@127.0.0.1:6379';
const CACHE_NAMESPACE = 'app-cache';
const DEFAULT_CACHE_TTL = 60000;
const redisUrl = process.env.REDIS_URL || DEFAULT_REDIS_URL;

const cache = new Keyv({
  store: new KeyvRedis(redisUrl),
  namespace: CACHE_NAMESPACE,
});

const handleCacheError = (error) => {
  console.error('Cache error:', error);
};

cache.on('error', handleCacheError);

/** Creates a cache-aside loader; cache errors are reported but never block loading. */
const createCacheAside =
  (
    cacheStore,
    defaultTtl = DEFAULT_CACHE_TTL,
    onCacheError = handleCacheError,
  ) =>
  async (key, loader, ttl = defaultTtl) => {
    try {
      const cachedValue = await cacheStore.get(key);
      if (cachedValue !== undefined) {
        return cachedValue;
      }
    } catch (error) {
      onCacheError(error);
    }

    const value = await loader();

    try {
      await cacheStore.set(key, value, ttl);
    } catch (error) {
      onCacheError(error);
    }

    return value;
  };

const getOrLoad = createCacheAside(cache);

export default cache;
export { cache, createCacheAside, DEFAULT_CACHE_TTL, getOrLoad };
