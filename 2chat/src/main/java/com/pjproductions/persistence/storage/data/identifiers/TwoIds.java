package com.pjproductions.persistence.storage.data.identifiers;

import com.pjproductions.persistence.storage.data.User;

public class TwoIds {
    private final long u1;
    private final long u2;

    public TwoIds(long u1, long u2) {
        if (u1 < u2){
            this.u1 = u1;
            this.u2 = u2;
        }
        else{
            this.u1 = u2;
            this.u2 = u1;
        }
    }

    public static TwoIds of(long u1, long u2){
        return new TwoIds(u1, u2);
    }

    public static TwoIds of(User  u1, User u2){
        return new TwoIds(u1.getId(), u2.getId());
    }

    public long small() {
        return u1;
    }

    public long big() {
        return u2;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (!(o instanceof TwoIds)) return false;

        TwoIds that = (TwoIds) o;

        if (u1 != that.u1) return false;
        return u2 == that.u2;
    }

    @Override
    public int hashCode() {
        int result = (int) (u1 ^ (u1 >>> 32));
        result = 31 * result + (int) (u2 ^ (u2 >>> 32));
        return result;
    }
}
