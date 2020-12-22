package com.pjproductions.persistence.storage.data;

import com.pjproductions.rest.exception.PersistenceException;

import java.util.concurrent.ConcurrentSkipListMap;

public class UserMessagesStub extends UserMessages {

    public UserMessagesStub() {
        super(new ConcurrentSkipListMap<>(), 0);
    }

    @Override
    public void addMessage(Message message) throws PersistenceException {

    }
}
