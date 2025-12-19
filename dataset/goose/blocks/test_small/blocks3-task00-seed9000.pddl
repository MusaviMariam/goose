(define (problem BW-3-9000-0)
    (:domain blocksworld)
    (:objects b1 b2 - block)
    (:init
        (handempty)
        (on-table b1)
        (on-table b2)
        (clear b1)
        (clear b2)
    )
    (:goal
        (and
            (on b2 b1)
        )
    )
)