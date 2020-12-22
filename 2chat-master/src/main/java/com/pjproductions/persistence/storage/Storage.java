package com.pjproductions.persistence.storage;

import com.fasterxml.jackson.annotation.JsonCreator;
import com.fasterxml.jackson.annotation.JsonProperty;
import com.pjproductions.persistence.storage.data.*;
import com.pjproductions.persistence.storage.data.identifiers.NamingId;
import com.pjproductions.persistence.storage.data.identifiers.NamingPool;
import com.pjproductions.persistence.storage.data.identifiers.TwoIds;
import com.pjproductions.persistence.storage.data.tofutureuse.ChannelMessage;
import com.pjproductions.persistence.storage.data.tofutureuse.Room;
import com.pjproductions.rest.definition.OperationResult;
import com.pjproductions.rest.definition.json.ReadMessages;
import com.pjproductions.rest.definition.json.ReadNewMessages;
import com.pjproductions.rest.exception.PersistenceException;

import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentMap;
import java.util.function.BiFunction;
import java.util.function.Function;
import java.util.stream.Collectors;

public class Storage {

    private final IdGenerator generatorIds;

    @JsonProperty("users-identifiers")
    private final NamingPool usersIdentifiers;

    @JsonProperty("users-by-name")
    private final ConcurrentHashMap<String, User> usersByName;

    @JsonProperty("user-by-email")
    private final ConcurrentHashMap<String, User> usersByEmail;

    @JsonProperty("user-by-Id")
    private final ConcurrentHashMap<Long, User> usersById;

    @JsonProperty("USERS")
    private final Set<User> users;

    @JsonProperty("user-affinities-by-id")
    private final ConcurrentHashMap<Long, UserAffinities> usersAffinities;

    @JsonProperty("messagesStub-pool-by-id")
    private final ConcurrentMap<TwoIds, UserMessages> messagesPool;

    @JsonProperty("messagesStub-for-each-user")
    private final ConcurrentMap<Long, ConcurrentMap<Long, UserMessages>> messagesPoolByUser;

    @JsonProperty("channel-messagesStub")
    private final ConcurrentHashMap<String, Messages<ChannelMessage>> messagesByChannel;

    @JsonProperty("channels")
    private final ConcurrentMap<String, Room> usersByChannel;

    @JsonCreator
    public Storage() {
        this(new ArrayList<>(), new ConcurrentHashMap<>(),
                new ConcurrentHashMap<>(), new ConcurrentHashMap<>(),
                new ConcurrentHashMap<>(),
                new ConcurrentHashMap<>(),
                new ConcurrentHashMap<>(),
                (long) 0);
    }

    @JsonCreator
    public Storage(Collection<User> users,
                   ConcurrentHashMap<Long, UserAffinities> usersAffinitiesId,
                   ConcurrentHashMap<TwoIds, UserMessages> messagesPoolId,
                   ConcurrentHashMap<String, Messages<ChannelMessage>> messagesByChannel,
                   ConcurrentHashMap<String, Room> usersByChannel,
                   ConcurrentMap<NamingId, Boolean> usersIdentifiers,
                   ConcurrentMap<Long, ConcurrentMap<Long, UserMessages>> messagesPoolByUser,
                   long lastIndex) {
        ConcurrentHashMap<User, User> map = new ConcurrentHashMap<>(users.parallelStream().collect(Collectors.toMap(Function.identity(), Function.identity())));
        this.users = map.newKeySet();
        this.usersByName = new ConcurrentHashMap<>(users.parallelStream().collect(Collectors.toMap(User::getUsername, Function.identity())));
        this.usersByEmail = new ConcurrentHashMap<>(users.parallelStream().collect(Collectors.toMap(User::getEmail, Function.identity())));
        this.usersById = new ConcurrentHashMap<>(users.parallelStream().collect(Collectors.toMap(User::getId, Function.identity())));
        this.usersAffinities = usersAffinitiesId;
        this.messagesPool = messagesPoolId;
        this.messagesByChannel = messagesByChannel;
        this.usersByChannel = usersByChannel;
        this.generatorIds = IdGenerator.defaultGenerator(lastIndex);
        this.usersIdentifiers = new NamingPool(usersIdentifiers);
        this.messagesPoolByUser = messagesPoolByUser;
    }

   public interface UsersOperations{
       User findUserByName(String username) throws PersistenceException;
       User findUserByEmail(String email) throws PersistenceException;
       User findUserByEmailOrName(String identifier) throws PersistenceException;
       User findUserById(Long id) throws PersistenceException;
       boolean usernameExists(String identifier);
       boolean emailExists(String email);
       User addUser(final String username, final String email, final String passHash) throws PersistenceException;
       void createUserResources(User u);
       void addFriend(User u, User friend) throws PersistenceException;
       UserAffinities getAffinities(User u);

       boolean isFriend(User u1, User u2);

       void removeFriend(User u1, User u2);

       void blockUser(User u1, User u2);

       boolean isBlocked(User u1, User u2);
   }

    public final UsersOperations USER_OPERATIONS = new UsersOperations (){

       public User findUserByName(String username){
           User u = usersByName.get(username);

           if(Objects.isNull(u)) throw new PersistenceException(OperationResult.NON_EXISTING_ACCOUNT);

           return u;
       }

       public User findUserByEmail(String email){
           User u = usersByEmail.get(email);

           if(Objects.isNull(u)) throw new PersistenceException(OperationResult.NON_EXISTING_ACCOUNT);

           return u;
       }

       public User findUserByEmailOrName(String identifier){
           User u = usersByName.get(identifier);

           if(!Objects.isNull(u)) return u;

           u = usersByEmail.get(identifier);

           if(!Objects.isNull(u)) return u;

           throw new PersistenceException(OperationResult.NON_EXISTING_ACCOUNT);
       }

       public User findUserById(Long id){
           User u = usersById.get(id);

           if(!Objects.isNull(u)) return u;

           throw new PersistenceException(OperationResult.NON_EXISTING_ACCOUNT);
       }

       public boolean usernameExists(String identifier){
           return usersByName.containsKey(identifier);
       }

       public boolean emailExists(String email){
           return usersByEmail.containsKey(email);
       }

       public User addUser(final String username, final String email, final String passHash) throws PersistenceException {
           usersIdentifiers.addUserIdentifier(username, email);

           long accountId = generatorIds.createId();

           User newUser = new User(accountId, username, email, passHash);

           createUserResources(newUser);

           usersByEmail.put(email, newUser);
           usersByName.put(username, newUser);
           usersById.put(accountId, newUser);

           return newUser;
       }

       public void createUserResources(User u){
           usersAffinities.put(u.getId(), new UserAffinities(u.getId()));
           messagesPoolByUser.put(u.getId(), new ConcurrentHashMap<>());
       }

        @Override
        public void addFriend(User u, User friend) throws PersistenceException {
            if(u.getId().equals(friend.getId())) throw new PersistenceException(OperationResult.INVALID_OPERATION);

            MESSAGES_OPERATIONS.createMessagesForTwoUsers(u, friend);
            UserAffinities affinities = getAffinities(u);
            affinities.createAffinity(friend);
        }

        @Override
        public UserAffinities getAffinities(User u) {
            return usersAffinities.get(u.getId());
        }

        @Override
        public boolean isFriend(User u1, User u2) {
            return getAffinities(u1).isFriend(u2);
        }

        @Override
        public void removeFriend(User u1, User u2) {
           UserAffinities affinities = getAffinities(u1);
           affinities.getFriends().remove(u2.getId());
           affinities.getSentRequests().remove(u2.getId());
        }

        @Override
        public void blockUser(User u1, User u2) {
            UserAffinities affinities = getAffinities(u1);

            Affinity aff = affinities.getAffinity(u2);
            if(aff == null){
                if(!affinities.isBlocked(u2)){
                    affinities.block(u2);
                }
                else{
                    affinities.unBlock(u2);
                }
            }
            else{
                affinities.blockFriend(u2, !aff.isBlocked());
            }
        }

        @Override
        public boolean isBlocked(User u1, User u2) {
            UserAffinities affinities = getAffinities(u1);

            return affinities.isBlockedUser(u2);
        }
    };

    public interface MessagesOperations{

        void createMessagesForTwoUsers(User u1, User u2);

        void sendMessage(User u1, User u2, Message msg);

        UserMessages messagesTwoUsers(User u1, User u2);

        <U, S> Map<S, Collection<U>> manyUsersMessages(User u1, Collection<ReadMessages<S>> users, Function<S, User> mapToUser, BiFunction<Long, Message, U> toU);

        <U> Collection<U> oneUserMessages(User u1, User friend, long from, long to, BiFunction<Long, Message, U> toU);

        <U, S> Map<S, Collection<U>> allMessages(User u, Function<Long, S> f, BiFunction<Long, Message, U> toU);

        <U, S> Map<S, Collection<U>> manyUsersNewMessages(User u1, Collection<ReadNewMessages<S>> users, Function<S, User> mapToUser, BiFunction<Long, Message, U> toU);
        <U> Collection<U> oneUserNewMessages(User u1, User friend, long from, BiFunction<Long, Message, U> toU);

        long lastIndex(User u1, User friend);
        Map<String, Long> lastIndex(User u1, Collection<String> friends);
        Map<String, Long> lastIndex(User u1);
    }

    public MessagesOperations MESSAGES_OPERATIONS = new MessagesOperations() {
        private final UserMessages messagesStub = new UserMessagesStub();

        @Override
        public void createMessagesForTwoUsers(User u1, User u2) {
            TwoIds key = TwoIds.of(u1, u2);
            messagesPool.putIfAbsent(key, new UserMessages());
            UserMessages messages = messagesPool.get(key);
            if(messages == null) throw new IllegalStateException("Problem in getting users");
            messagesPoolByUser.putIfAbsent(u1.getId(), new ConcurrentHashMap<>());
            messagesPoolByUser.get(u1.getId()).putIfAbsent(u2.getId(), messages);
        }

        @Override
        public void sendMessage(User u1, User u2, Message msg) {
            messagesTwoUsers(u1, u2).addMessage(msg);
        }

        @Override
        public UserMessages messagesTwoUsers(User u1, User u2) {
            TwoIds key = TwoIds.of(u1, u2);
            return messagesPool.getOrDefault(key, messagesStub);
        }

        @Override
        public <U, S> Map<S, Collection<U>> manyUsersMessages(User u1, Collection<ReadMessages<S>> users, Function<S, User> mapToUser, BiFunction<Long, Message, U> toU) {
            final ConcurrentMap<Long, UserMessages> messages = messagesPoolByUser.get(u1.getId());
            return users.stream()
                    .map(r -> new AbstractMap.SimpleEntry<>(r, mapToUser.apply(r.getFriend())))
                    .map(r -> {
                        final UserMessages m = messages.get(r.getValue().getId());
                        if(Objects.isNull(m)) throw new PersistenceException(OperationResult.INVALID_OPERATION);
                        return new AbstractMap.SimpleEntry<>(r.getKey().getFriend(), m.mapTo(toU, r.getKey().getTo(), r.getKey().getFrom()));
                    })
                    .collect(Collectors.toMap(Map.Entry::getKey, Map.Entry::getValue));
        }

        @Override
        public <U> Collection<U> oneUserMessages(User u1, User friend, long from, long to, BiFunction<Long, Message, U> toU) {
            UserMessages response = messagesPoolByUser.get(u1.getId()).get(friend.getId());
            return Objects.isNull(response) ? null : response.mapTo(toU, to, from);
        }

        @Override
        public <U, S> Map<S, Collection<U>> allMessages(User u, Function<Long, S> f, BiFunction<Long, Message, U> toU) {
            final ConcurrentMap<Long, UserMessages> messages = messagesPoolByUser.get(u.getId());

            return messages.entrySet().stream()
                    .map(r -> new AbstractMap.SimpleEntry<>(f.apply(r.getKey()), r.getValue().mapTo(toU)))
                    .collect(Collectors.toMap(Map.Entry::getKey, Map.Entry::getValue));
        }

        @Override
        public <U, S> Map<S, Collection<U>> manyUsersNewMessages(User u1, Collection<ReadNewMessages<S>> users, Function<S, User> mapToUser, BiFunction<Long, Message, U> toU) {
            final ConcurrentMap<Long, UserMessages> messages = messagesPoolByUser.get(u1.getId());

            return users.stream()
                    .map(r -> new AbstractMap.SimpleEntry<>(r, mapToUser.apply(r.friend())))
                    .map(r -> {
                        final UserMessages m = messages.get(r.getValue().getId());
                        if(Objects.isNull(m)) throw new PersistenceException(OperationResult.INVALID_OPERATION);
                        return new AbstractMap.SimpleEntry<>(r.getKey().friend(), m.mapTo(toU, r.getKey().fromIndex()));
                    })
                    .collect(Collectors.toMap(Map.Entry::getKey, Map.Entry::getValue));

        }

        @Override
        public <U> Collection<U> oneUserNewMessages(User u1, User friend, long from, BiFunction<Long, Message, U> toU) {
            UserMessages response = messagesPoolByUser.get(u1.getId()).get(friend.getId());

            return Objects.isNull(response) ? null : response.mapTo(toU, from);
        }

        @Override
        public long lastIndex(User u1, User friend) {
            UserMessages messages = messagesPoolByUser.get(u1.getId()).get(friend.getId());
            if(Objects.isNull(messages)){
                throw new PersistenceException(OperationResult.INVALID_OPERATION);
            }

            return messages.lastIndex();
        }

        @Override
        public Map<String, Long> lastIndex(User u1, Collection<String> friends) {
            final ConcurrentMap<Long, UserMessages> messages = messagesPoolByUser.get(u1.getId());
            return friends.stream()
                    .map(s -> new AbstractMap.SimpleEntry<>(s, USER_OPERATIONS.findUserByEmailOrName(s)))
                    .map(e -> {
                        final UserMessages m = messages.get(e.getValue().getId());
                        if(Objects.isNull(m)) throw new PersistenceException(OperationResult.INVALID_OPERATION);
                        return new AbstractMap.SimpleEntry<>(e.getKey(), m.lastIndex());
                    })
                    .collect(Collectors.toMap(Map.Entry::getKey, Map.Entry::getValue));
        }

        @Override
        public Map<String, Long> lastIndex(User u1) {
            return messagesPoolByUser.get(u1.getId())
                    .entrySet()
                    .stream()
                    .map(e -> {
                        User u = USER_OPERATIONS.findUserById(e.getKey());
                        return new AbstractMap.SimpleEntry<>(u.getUsername(), e.getValue().lastIndex());
                    })
                    .collect(Collectors.toMap(Map.Entry::getKey, Map.Entry::getValue));
        }
    };




}
