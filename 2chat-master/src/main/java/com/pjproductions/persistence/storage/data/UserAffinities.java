package com.pjproductions.persistence.storage.data;

import com.fasterxml.jackson.annotation.JsonProperty;
import com.pjproductions.persistence.storage.data.tofutureuse.Room;
import com.pjproductions.rest.definition.OperationResult;
import com.pjproductions.rest.exception.PersistenceException;

import java.util.Collection;
import java.util.HashSet;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentMap;
import java.util.function.Function;
import java.util.stream.Collectors;

public class UserAffinities {

    @JsonProperty("id")
    private final Long id;

    @JsonProperty("friends")
    private final ConcurrentMap<Long, Affinity> friends;

    @JsonProperty("blocked-USERS")
    private final Set<Long> blockedUsers;

    @JsonProperty("current-channel")
    private Room channel;

    @JsonProperty("requests")
    private final Set<Long> requests;

    @JsonProperty("sent-requests")
    private final Set<Long> sentRequests;

    public UserAffinities(Long id) {
        this(id, new HashSet<>(), new HashSet<>(), new HashSet<>());
    }

    public UserAffinities(Long id, Set<Affinity> friends, Set<Room> channels, Set<Long> blockedUsers) {
        this.id = id;
        this.friends = new ConcurrentHashMap<>(friends.parallelStream().collect(Collectors.toMap(Affinity::getUserId, Function.identity())));
        ConcurrentHashMap<Long,Long> map = new ConcurrentHashMap<>();
        requests = map.newKeySet();
        map = new ConcurrentHashMap<>();
        sentRequests = map.newKeySet();
        map = new ConcurrentHashMap<>();
        this.blockedUsers = map.newKeySet();
        this.blockedUsers.addAll(blockedUsers);
    }

    public Long getId() {
        return id;
    }

    public Affinity getAffinity(User friend) {
        return friends.get(friend.getId());
    }

    public void createAffinity(User friend){
        friends.putIfAbsent(friend.getId(), new Affinity(friend.getId(), false));
    }

    public boolean isFriend(User friend){
        return friends.containsKey(friend.getId());
    }

    public boolean isBlocked(User friend){
        return blockedUsers.contains(friend.getId());
    }

    public boolean isBlockedUser(User friend){
        return !((isFriend(friend) && !getAffinity(friend).isBlocked()) || !isBlocked(friend));
    }

    public void block(User friend){
        blockedUsers.add(friend.getId());
    }

    public void unBlock(User friend){
        blockedUsers.remove(friend.getId());
    }

    public void blockFriend(User friend, boolean blocked){
        friends.put(friend.getId(), new Affinity(friend.getId(), blocked));
    }

    public void addFriendRequest(User friend){
        if(requests.contains(friend.getId())) throw new PersistenceException(OperationResult.DUPLICATED_OPERATION);
        if(isFriend(friend) || isBlocked(friend)) throw new PersistenceException(OperationResult.ILLEGAL_ACTION);
        requests.add(friend.getId());
    }

    public void addSentRequest(User friend){
        if(isFriend(friend)) throw new PersistenceException(OperationResult.ILLEGAL_ACTION);
        sentRequests.add(friend.getId());
    }

    public void promoteRequestToFriend(User friend, boolean accept) throws PersistenceException {
        boolean res = requests.remove(friend.getId());
        if(!res) throw new PersistenceException(OperationResult.INVALID_OPERATION);

        if(accept) {
            createAffinity(friend);
        }
    }

    public void requestAccepted(User friend) {
        boolean res = sentRequests.remove(friend.getId());
        if(res){
            createAffinity(friend);
        }

        if(isBlocked(friend)) unBlock(friend);
    }

    public Room getChannel() {
        return channel;
    }
    public void setChannel(Room channel) {
        this.channel = channel;
    }

    public ConcurrentMap<Long, Affinity> getFriends() {
        return friends;
    }

    public Set<Long> getBlockedUsers() {
        return blockedUsers;
    }

    public Set<Long> getRequests() {
        return requests;
    }

    public Set<Long> getSentRequests() {
        return sentRequests;
    }

    public <T> Collection<T> mapFriend(Function<Affinity, T> map){
        return friends.values().stream()
                .map(map)
                .collect(Collectors.toList());
    }

    public <T> Collection<T> mapSentRequests(Function<Long, T> map){
        return sentRequests.stream()
                .map(map)
                .collect(Collectors.toList());
    }

    public <T> Collection<T> mapFriendRequests(Function<Long, T> map){
        return requests.stream().map(map).collect(Collectors.toList());
    }
}
