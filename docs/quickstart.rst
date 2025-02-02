Quickstart guide
================

Here is a simple example to get you started with the Spring library:

.. code-block:: python

    from twdspring import spring

    # Define query object
    query = np.array((11, 6, 9, 4))

    # Create a new Spring object
    spring = Spring(query_vector=query, epsilon=15, use_z_norm=False)

    x = [5, 6, 12, 6, 10, 6, 5, 13]
    results = []
    search_gen = spring.search()
    next(search_gen)
    results = (search_gen.send(val) for val in x)

    print(list(dropwhile(lambda x: not x.status, results)))

The result of this code will be:

.. code-block:: python

    [
        Searcher(status='tracking', twd_min=14.0, t_start=2, t_end=5, t=4),
        Searcher(status='tracking', twd_min=14.0, t_start=2, t_end=5, t=4),
        Searcher(status='tracking', twd_min=6.0, t_start=2, t_end=7, t=6),
        Searcher(status='tracking', twd_min=6.0, t_start=2, t_end=7, t=6),
        Searcher(status='match', twd_min=6.0, t_start=2, t_end=7, t=6)
    ]

You can use Spring with z-normalization. It will use moving avarage and standard deviation to normalize the data.
