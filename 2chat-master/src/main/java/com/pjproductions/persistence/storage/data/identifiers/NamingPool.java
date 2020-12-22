package com.pjproductions.persistence.storage.data.identifiers;

import com.pjproductions.rest.definition.OperationResult;
import com.pjproductions.rest.exception.PersistenceException;

import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentMap;

public class NamingPool {
    private final ConcurrentMap<NamingId, Boolean> identifiers;

    public NamingPool(ConcurrentMap<NamingId, Boolean> identifiers) {
        this.identifiers = identifiers;
    }

    public NamingPool(){
        identifiers = new ConcurrentHashMap<>();
    }

    public void addUserIdentifier(String username, String email) throws PersistenceException {

        NamingId nameId = NamingId.ofEmail(email);
        Boolean existEmail = identifiers.putIfAbsent(nameId, true);
        if(existEmail != null) throw new PersistenceException(OperationResult.EXISTING_ACCOUNT);

        Boolean existUsername = identifiers.putIfAbsent(NamingId.ofUsername(username), true);
        if(existUsername != null) {
            identifiers.remove(nameId, true);
            throw new PersistenceException(OperationResult.EXISTING_USERNAME);
        }
    }

}

