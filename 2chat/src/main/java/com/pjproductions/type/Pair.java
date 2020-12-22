package com.pjproductions.type;

public class Pair<T,U> {
    private final T t;
    private final U u;

    public Pair(T t, U u) {
        this.t = t;
        this.u = u;
    }

    public static<T,U> Pair<T,U> of(T t, U u){
        return new Pair<>(t, u);
    }

    public T getFirst() {
        return t;
    }

    public U getSecond() {
        return u;
    }
}
