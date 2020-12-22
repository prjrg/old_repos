package com.pjproductions.rosalimao.model.images;

import com.pjproductions.rosalimao.model.images.item.Image;
import org.springframework.data.repository.PagingAndSortingRepository;

public interface PictureRepository extends PagingAndSortingRepository<Image, Long> {
}
