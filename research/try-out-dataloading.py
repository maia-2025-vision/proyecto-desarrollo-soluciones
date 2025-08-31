import marimo

__generated_with = "0.15.2"
app = marimo.App(width="medium")


@app.cell
def _():
    from pathlib import Path
    import os
    print(os.getcwd())
    return (Path,)


@app.cell
def _(Path):

    from cow_detect.train.teo.ds_v1 import SkyDataset
    from cow_detect.utils.debug import summarize

    # help(SkyDataset)

    image_paths = list(Path("data/sky/Dataset1/").rglob("*.JPG"))[10: 13]
    ds = SkyDataset(
        name="test", 
        root_dir=Path("data/sky/Dataset1/"), 
        image_paths=image_paths
    )
    return ds, summarize


@app.cell
def _(ds, summarize):
    for i in range(0, 3):
        print(i, summarize(ds[i]))
    return


@app.cell
def _():
    from torch.utils.data import DataLoader
    return (DataLoader,)


@app.cell
def _(DataLoader, ds, summarize):
    def custom_collate(batch): 
        print(f"custom collate starts:\n{summarize(batch)}")
        return tuple(zip(*batch, strict=True)) 
    
    dl = DataLoader(dataset=ds, batch_size=3, collate_fn=custom_collate)
    return (dl,)


@app.cell
def _(dl, summarize):
    for batch in dl: 
        print(summarize(batch))

    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
