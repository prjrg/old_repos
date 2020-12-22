package com.pjproductions.persistence.storage.data;

import java.util.concurrent.atomic.AtomicLong;

public interface IdGenerator {

        long createId();

        long getId();

        IdGenerator setId(long id);

    static IdGenerator defaultGenerator(long init){
        return new IdGenerator() {
            private final AtomicLong value = new AtomicLong(init);

            @Override
            public long createId() {
                return value.incrementAndGet();
            }

            @Override
            public long getId() {
                return value.get();
            }

            @Override
            public IdGenerator setId(long id) {
                value.set(id);
                return this;
            }
        };
    }
}
