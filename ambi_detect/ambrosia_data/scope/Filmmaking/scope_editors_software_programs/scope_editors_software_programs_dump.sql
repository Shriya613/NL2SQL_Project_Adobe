BEGIN TRANSACTION;
CREATE TABLE Actors (
    actor_id INTEGER PRIMARY KEY AUTOINCREMENT,
    first_name TEXT,
    last_name TEXT
);
INSERT INTO "Actors" VALUES(1,'Andy','Dufraine');
INSERT INTO "Actors" VALUES(2,'Tom','Hanks');
INSERT INTO "Actors" VALUES(3,'Leonardo','DiCaprio');
INSERT INTO "Actors" VALUES(4,'Elijah','Wood');
INSERT INTO "Actors" VALUES(5,'Kevin','Costner');
CREATE TABLE DirectorFilmProjects (
    director_film_project_id INTEGER PRIMARY KEY AUTOINCREMENT,
    director_id INTEGER REFERENCES Directors(director_id),
    film_project_id INTEGER REFERENCES FilmProjects(project_id)
);
INSERT INTO "DirectorFilmProjects" VALUES(1,1,1);
INSERT INTO "DirectorFilmProjects" VALUES(2,2,NULL);
INSERT INTO "DirectorFilmProjects" VALUES(3,3,3);
INSERT INTO "DirectorFilmProjects" VALUES(4,4,NULL);
INSERT INTO "DirectorFilmProjects" VALUES(5,5,NULL);
CREATE TABLE Directors (
    director_id INTEGER PRIMARY KEY AUTOINCREMENT,
    first_name TEXT,
    last_name TEXT
);
INSERT INTO "Directors" VALUES(1,'Frank','Capra');
INSERT INTO "Directors" VALUES(2,'Martin','Scorsese');
INSERT INTO "Directors" VALUES(3,'Quentin','Tarantino');
INSERT INTO "Directors" VALUES(4,'Steven','Spielberg');
INSERT INTO "Directors" VALUES(5,'Ronald','Reagan');
CREATE TABLE EditorDirectorRelationship (
    relationship_id INTEGER PRIMARY KEY AUTOINCREMENT,
    director_id INTEGER REFERENCES Directors(director_id),
    editor_id INTEGER REFERENCES Editors(editor_id)
);
INSERT INTO "EditorDirectorRelationship" VALUES(1,1,1);
INSERT INTO "EditorDirectorRelationship" VALUES(2,2,2);
INSERT INTO "EditorDirectorRelationship" VALUES(3,3,3);
INSERT INTO "EditorDirectorRelationship" VALUES(4,4,4);
INSERT INTO "EditorDirectorRelationship" VALUES(5,5,5);
CREATE TABLE EditorSoftwarePrograms (
    editor_software_program_id INTEGER PRIMARY KEY AUTOINCREMENT,
    editor_id INTEGER REFERENCES Editors(editor_id),
    software_program_id INTEGER REFERENCES Software_Programs(program_id)
);
INSERT INTO "EditorSoftwarePrograms" VALUES(1,1,1);
INSERT INTO "EditorSoftwarePrograms" VALUES(2,2,2);
INSERT INTO "EditorSoftwarePrograms" VALUES(3,3,3);
INSERT INTO "EditorSoftwarePrograms" VALUES(4,4,4);
INSERT INTO "EditorSoftwarePrograms" VALUES(5,5,5);
INSERT INTO "EditorSoftwarePrograms" VALUES(6,2,1);
INSERT INTO "EditorSoftwarePrograms" VALUES(7,3,1);
INSERT INTO "EditorSoftwarePrograms" VALUES(8,4,1);
INSERT INTO "EditorSoftwarePrograms" VALUES(9,5,1);
CREATE TABLE Editors (
    editor_id INTEGER PRIMARY KEY AUTOINCREMENT,
    full_name TEXT);
INSERT INTO "Editors" VALUES(1,'John Doe');
INSERT INTO "Editors" VALUES(2,'Jane Smith');
INSERT INTO "Editors" VALUES(3,'Michael Brown');
INSERT INTO "Editors" VALUES(4,'Emma White');
INSERT INTO "Editors" VALUES(5,'David Black');
CREATE TABLE FilmProjects (
    project_id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT,
    release_date DATE
);
INSERT INTO "FilmProjects" VALUES(1,'The Shawshank Redemption','1994-09-22');
INSERT INTO "FilmProjects" VALUES(2,'The Godfather','1972-03-24');
INSERT INTO "FilmProjects" VALUES(3,'Pulp Fiction','1994-10-14');
INSERT INTO "FilmProjects" VALUES(4,'Schindler''s List','1993-12-15');
INSERT INTO "FilmProjects" VALUES(5,'Forrest Gump','1994-07-06');
CREATE TABLE ProjectActors (
    project_actor_id INTEGER PRIMARY KEY AUTOINCREMENT,
    film_project_id INTEGER REFERENCES FilmProjects(project_id),
    actor_id INTEGER REFERENCES Actors(actor_id)
);
INSERT INTO "ProjectActors" VALUES(1,1,1);
INSERT INTO "ProjectActors" VALUES(2,1,2);
INSERT INTO "ProjectActors" VALUES(3,2,NULL);
INSERT INTO "ProjectActors" VALUES(4,3,NULL);
INSERT INTO "ProjectActors" VALUES(5,4,NULL);
CREATE TABLE Software_Programs (
    program_id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    version TEXT
);
INSERT INTO "Software_Programs" VALUES(1,'Adobe Premiere','2021');
INSERT INTO "Software_Programs" VALUES(2,'Final Cut Pro','2020');
INSERT INTO "Software_Programs" VALUES(3,'DaVinci Resolve','16');
INSERT INTO "Software_Programs" VALUES(4,'Avid Media Composer','2020');
INSERT INTO "Software_Programs" VALUES(5,'HitFilm Express','13');
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Editors',5);
INSERT INTO "sqlite_sequence" VALUES('Software_Programs',5);
INSERT INTO "sqlite_sequence" VALUES('FilmProjects',5);
INSERT INTO "sqlite_sequence" VALUES('Directors',5);
INSERT INTO "sqlite_sequence" VALUES('Actors',5);
INSERT INTO "sqlite_sequence" VALUES('EditorDirectorRelationship',5);
INSERT INTO "sqlite_sequence" VALUES('ProjectActors',5);
INSERT INTO "sqlite_sequence" VALUES('DirectorFilmProjects',5);
INSERT INTO "sqlite_sequence" VALUES('EditorSoftwarePrograms',9);
COMMIT;
