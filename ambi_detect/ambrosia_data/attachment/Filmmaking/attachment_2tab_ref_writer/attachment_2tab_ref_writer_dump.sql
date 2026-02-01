BEGIN TRANSACTION;
CREATE TABLE Education(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    institution TEXT
);
INSERT INTO "Education" VALUES(1,'New York University');
INSERT INTO "Education" VALUES(2,'University of Southern California');
INSERT INTO "Education" VALUES(3,'American Film Institute');
INSERT INTO "Education" VALUES(4,'National Film and Television School');
INSERT INTO "Education" VALUES(5,'London Film Academy');
CREATE TABLE Film_Director(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    birthdate DATE,
    nationality TEXT,
    educationId INTEGER,
    FOREIGN KEY (educationId) REFERENCES Education(id)
);
INSERT INTO "Film_Director" VALUES(1,'Tom Miller','1985-09-21','USA',1);
INSERT INTO "Film_Director" VALUES(2,'Samantha Wilson','1991-11-11','UK',2);
INSERT INTO "Film_Director" VALUES(3,'Steven Davis','1992-03-24','Canada',3);
INSERT INTO "Film_Director" VALUES(4,'Kevin Wright','1989-02-01','Australia',4);
INSERT INTO "Film_Director" VALUES(5,'Robert Young','1993-04-12','South Africa',5);
CREATE TABLE Playwright(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    birthdate DATE,
    nationality TEXT,
    educationId INTEGER,
    FOREIGN KEY (educationId) REFERENCES Education(id)
);
INSERT INTO "Playwright" VALUES(1,'Emma Green','1990-04-15','USA',1);
INSERT INTO "Playwright" VALUES(2,'David Black','1982-08-12','UK',2);
INSERT INTO "Playwright" VALUES(3,'Michael Gray','1987-12-03','Canada',3);
INSERT INTO "Playwright" VALUES(4,'Laura Red','1993-05-18','Australia',4);
INSERT INTO "Playwright" VALUES(5,'William Blue','1994-06-04','South Africa',5);
CREATE TABLE Screenwriter(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    birthdate DATE,
    nationality TEXT,
    educationId INTEGER,
    FOREIGN KEY (educationId) REFERENCES Education(id)
);
INSERT INTO "Screenwriter" VALUES(1,'John Doe','1980-06-23','USA',1);
INSERT INTO "Screenwriter" VALUES(2,'Jane Smith','1978-09-10','UK',2);
INSERT INTO "Screenwriter" VALUES(3,'Alice Brown','1992-11-05','Canada',3);
INSERT INTO "Screenwriter" VALUES(4,'Bob Johnson','1995-02-22','Australia',4);
INSERT INTO "Screenwriter" VALUES(5,'Charlie White','1988-03-07','South Africa',5);
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Education',5);
INSERT INTO "sqlite_sequence" VALUES('Screenwriter',5);
INSERT INTO "sqlite_sequence" VALUES('Playwright',5);
INSERT INTO "sqlite_sequence" VALUES('Film_Director',5);
COMMIT;
