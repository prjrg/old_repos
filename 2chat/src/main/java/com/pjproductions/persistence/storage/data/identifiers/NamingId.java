package com.pjproductions.persistence.storage.data.identifiers;

public class NamingId {
    private final KeyIdentifier identifier;
    private final String value;

    public NamingId(KeyIdentifier identifier, String value) {
        this.identifier = identifier;
        this.value = value;
    }

    public static NamingId ofUsername(String username){
        return new NamingId(KeyIdentifier.USERNAME, username);
    }

    public static NamingId ofEmail(String email){
        return new NamingId(KeyIdentifier.EMAIL, email);
    }

    public KeyIdentifier getIdentifier() {
        return identifier;
    }

    public String getValue() {
        return value;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (!(o instanceof NamingId)) return false;

        NamingId namingId = (NamingId) o;

        if (getIdentifier() != namingId.getIdentifier()) return false;
        return getValue().equals(namingId.getValue());
    }

    @Override
    public int hashCode() {
        int result = getIdentifier().hashCode();
        result = 31 * result + getValue().hashCode();
        return result;
    }
}