package com.pjproductions.rosalimao.model.images;

import com.pjproductions.rosalimao.model.images.item.Model;
import org.springframework.data.jpa.repository.Query;
import org.springframework.data.repository.PagingAndSortingRepository;

import java.util.List;

public interface ImageModelRepository extends PagingAndSortingRepository<Model, Long> {

    @Query("SELECT d.id FROM Image d where d.model=?1")
    List<Long> queryByModel(Model model);

}
