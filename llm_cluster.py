import click
import json
import llm
import numpy as np
import sklearn.cluster
import sqlite_utils
import textwrap

DEFAULT_SUMMARY_PROMPT = """
Short, concise title for this cluster of related documents.
""".strip()

@llm.hookimpl
def register_commands(cli):
    @cli.command()
    @click.argument("collection")
    @click.argument("n", type=float)
    @click.option(
        "--truncate",
        type=int,
        default=100,
        help="Truncate content to this many characters - 0 for no truncation",
    )
    @click.option(
        "-d",
        "--database",
        type=click.Path(
            file_okay=True, allow_dash=False, dir_okay=False, writable=True
        ),
        envvar="LLM_EMBEDDINGS_DB",
        help="SQLite database file containing embeddings",
    )
    @click.option(
        "--summary", is_flag=True, help="Generate summary title for each cluster"
    )
    @click.option("-m", "--model", help="LLM model to use for the summary")
    @click.option("--prompt", help="Custom prompt to use for the summary")
    @click.option("-a", "--algorithm", type=click.Choice(['kmeans', 'dbscan'], case_sensitive=False), default='kmeans', help="Clustering algorithm to use ('kmeans' or 'dbscan').")
    def cluster(collection, n, truncate, database, summary, model, prompt, algorithm):
        """
        Generate clusters from embeddings in a collection.

        For DBSCAN, the 'n' parameter refers to the 'eps' value. For effective tuning of 'eps',
        it's recommended to start with a value based on the distance metric of your dataset.
        Example usage:
        - For DBSCAN: llm cluster my_collection 0.5 --algorithm dbscan
        - For KMeans: llm cluster my_collection 5 --algorithm kmeans
        """

        from llm.cli import get_default_model, get_key

        if algorithm.lower() == 'dbscan':
            # Note: For DBSCAN, 'n' is used as the 'eps' parameter.
            clustering_model = sklearn.cluster.DBSCAN(eps=n, min_samples=3)
        elif algorithm.lower() == 'kmeans':
            # For KMeans, 'n' needs to be an integer specifying the number of clusters.
            if not n.is_integer():
                raise ValueError("For KMeans, the 'n' parameter must be an integer specifying the number of clusters.")
            clustering_model = sklearn.cluster.MiniBatchKMeans(n_clusters=int(n))
        
        if database:
            db = sqlite_utils.Database(database)
        else:
            db = sqlite_utils.Database(llm.user_dir() / "embeddings.db")
        
        rows = [
            (row[0], llm.decode(row[1]), row[2])
            for row in db.execute(
                """
            select id, embedding, content from embeddings
            where collection_id = (
                select id from collections where name = ?
            )
            """,
                [collection],
            ).fetchall()
        ]
        to_cluster = np.array([item[1] for item in rows])
        clustering_model.fit(to_cluster)
        assignments = clustering_model.labels_

        # Helper function for text truncation
        def truncate_text(text):
            if not text:
                return None
            if truncate > 0:
                return text[:truncate]
            else:
                return text

        # Organizing clusters
        clusters = {}
        for (id, _, content), cluster in zip(rows, assignments):
            clusters.setdefault(str(cluster), []).append(
                {"id": str(id), "content": truncate_text(content)}
            )

        output_clusters = [{"id": k, "items": v} for k, v in clusters.items()]

        # Generating summaries if requested
        if summary:
            model = llm.get_model(model or get_default_model())
            if model.needs_key:
                model.key = get_key("", model.needs_key, model.key_env_var)
            prompt = prompt or DEFAULT_SUMMARY_PROMPT
            click.echo("[")
            for cluster, is_last in zip(
                output_clusters, [False] * (len(output_clusters) - 1) + [True]
            ):
                click.echo("  {")
                click.echo('    "id": {},'.format(json.dumps(cluster["id"])))
                click.echo(
                    '    "items": '
                    + textwrap.indent(
                        json.dumps(cluster["items"], indent=2), "    "
                    ).lstrip()
                    + ","
                )
                prompt_content = "\n".join(
                    [item["content"] for item in cluster["items"] if item["content"]]
                )
                if prompt_content.strip():
                    summary = model.prompt(
                        prompt_content,
                        system=prompt,
                    ).text()
                else:
                    summary = None
                click.echo('    "summary": {}'.format(json.dumps(summary)))
                click.echo("  }" + ("," if not is_last else ""))
            click.echo("]")
        else:
            click.echo(json.dumps(output_clusters, indent=4))

