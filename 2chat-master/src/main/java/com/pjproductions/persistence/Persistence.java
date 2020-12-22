package com.pjproductions.persistence;

import com.pjproductions.persistence.controllers.MessengerDataController;
import com.pjproductions.persistence.controllers.UserDataController;
import com.pjproductions.persistence.controllers.impl.MessengerDataControllerImpl;
import com.pjproductions.persistence.controllers.impl.UserDataControllerImp;
import com.pjproductions.persistence.storage.Storage;

public final class Persistence {
    public final UserDataController USERS_CONTROLLER;
    public final MessengerDataController MESSENGER_CONTROLLER;

    private final Storage storage;

    private static class LazyHolder {
        static final Persistence INSTANCE = new Persistence();
    }

    private Persistence(){
        storage = new Storage();
        USERS_CONTROLLER = new UserDataControllerImp(storage);
        MESSENGER_CONTROLLER = new MessengerDataControllerImpl(storage);
    }

    public static Persistence manager() {
        return LazyHolder.INSTANCE;
    }
}
