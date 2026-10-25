"""Create a demo project with a species list and an empty dataset (run from backend/, after migrations).

    cd backend && python ../scripts/seed_demo.py --species "Panthera onca" "Tapirus bairdii" "Pecari tajacu"
Create users with `python -m app.cli create-user ...` first.
"""

import argparse

from sqlalchemy import select

from app.db.session import session_factory
from app.models import Dataset, Project, ProjectSpecies, Taxon

DEFAULT_SPECIES = ["Panthera onca", "Puma concolor", "Tapirus bairdii", "Pecari tajacu", "Dasyprocta punctata",
                   "Leopardus pardalis", "Nasua narica", "Cuniculus paca"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="Demo: Osa-style camera traps")
    parser.add_argument("--species", nargs="*", default=DEFAULT_SPECIES)
    args = parser.parse_args()
    with session_factory()() as db:
        project = Project(name=args.name, description="Seeded demo project",
                          settings={"alpha": 0.05, "review_budget": 0.1, "country": "CRI"})
        db.add(project)
        db.flush()
        for name in args.species:
            taxon = db.execute(select(Taxon).where(Taxon.scientific_name == name)).scalar_one_or_none()
            if taxon is None:
                taxon = Taxon(scientific_name=name)
                db.add(taxon)
                db.flush()
            db.add(ProjectSpecies(project_id=project.id, taxon_id=taxon.id))
        db.add(Dataset(project_id=project.id, name="demo-upload", license="see dataset card"))
        db.commit()
        print(f"project {project.id} created with {len(args.species)} species")


if __name__ == "__main__":
    main()
