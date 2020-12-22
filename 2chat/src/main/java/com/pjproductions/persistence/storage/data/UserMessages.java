package com.pjproductions.persistence.storage.data;

import com.pjproductions.rest.definition.OperationResult;
import com.pjproductions.rest.exception.PersistenceException;

import java.util.Collection;
import java.util.Map;
import java.util.concurrent.ConcurrentSkipListMap;
import java.util.function.BiFunction;
import java.util.stream.Collectors;

public class UserMessages extends Messages<Message> {

    public UserMessages(ConcurrentSkipListMap<Long, Message> messagesById, long indexCounter) {
        super(messagesById, indexCounter);
    }

    public UserMessages(){
        super();
    }

    public void addMessage(Message message) throws PersistenceException {
        long index = idGenerator.createId();
        message.setId(index);
        messagesById.put(index, message);
    }

    public <T> Collection<T> mapTo(BiFunction<Long, Message, T> f){
        return messagesById.entrySet().stream()
                .map(e -> f.apply(e.getKey(), e.getValue()))
                .collect(Collectors.toList());
    }

    public <T> Collection<T> mapTo(BiFunction<Long, Message, T> f, long first, long last){
        if(first > last || last < 0) throw new PersistenceException(OperationResult.INVALID_OPERATION);
        try {
            Map<Long, Message> view = messagesById.subMap(first, true, last, true);
            return view.entrySet().stream()
                    .map(e -> f.apply(e.getKey(), e.getValue()))
                    .collect(Collectors.toList());
        }
        catch(IllegalArgumentException e){
            System.out.println("Missed indices");
            throw new PersistenceException(OperationResult.INVALID_OPERATION);
        }

    }

    public <T> Collection<T> mapTo(BiFunction<Long, Message, T> f, long index){
        if(index <= 0) throw new PersistenceException(OperationResult.INVALID_OPERATION);
        try {
            Map<Long, Message> view = messagesById.tailMap(index, true);
            return view.entrySet().stream()
                    .map(e -> f.apply(e.getKey(), e.getValue()))
                    .collect(Collectors.toList());
        }
        catch(IllegalArgumentException e){
            System.out.println("Missed indices");
            throw new PersistenceException(OperationResult.INVALID_OPERATION);
        }
    }

    public long lastIndex(){
        return idGenerator.getId();
    }

}
