package com.pjproductions.rosalimao.model.images.item;

import lombok.ToString;

import javax.persistence.*;

@Entity
@Table(name = "picture")
public class Image {

    @Id
    @Column(name = "id")
    @GeneratedValue(strategy = GenerationType.SEQUENCE)
    private long id;

    @Column(name = "name")
    private String name;

    @Column(name = "type")
    private String type;

    @Column(name = "picture")
    @Lob
    private byte[] picture;

    @ManyToOne
    @JoinColumn(name="model_id", nullable=false)
    @ToString.Exclude
    private Model model;

    public Image() {
    }

    public Image(String name, String type, byte[] picture, Model model) {
        this.name = name;
        this.type = type;
        this.picture = picture;
        this.model = model;
    }

    public long getId() {
        return id;
    }

    public void setId(long id) {
        this.id = id;
    }

    public String getName() {
        return name;
    }

    public void setName(String name) {
        this.name = name;
    }

    public String getType() {
        return type;
    }

    public void setType(String type) {
        this.type = type;
    }

    public byte[] getPicture() {
        return picture;
    }

    public void setPicture(byte[] picture) {
        this.picture = picture;
    }

    public Model getModel() {
        return model;
    }

    public void setModel(Model model) {
        this.model = model;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;

        Image image = (Image) o;

        return id == image.id;
    }

    @Override
    public int hashCode() {
        return (int) (id ^ (id >>> 32));
    }
}
