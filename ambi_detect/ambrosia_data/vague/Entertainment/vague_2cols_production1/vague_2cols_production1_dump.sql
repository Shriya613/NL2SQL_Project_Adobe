BEGIN TRANSACTION;
CREATE TABLE Actor (
    id INTEGER PRIMARY KEY,
    name TEXT,
    dateOfBirth DATE,
    gender TEXT,
    role TEXT);
INSERT INTO "Actor" VALUES(1,'Kevin Spacey','July 26, 1959','Male','Richard III');
INSERT INTO "Actor" VALUES(2,'Cate Blanchett','May 14, 1969','Female','Lady Macbeth');
CREATE TABLE "Stage_Play" (
    id INTEGER PRIMARY KEY,
    title TEXT,
    year_premiered INT,
    genre TEXT,
    `theater_company` TEXT, `producer` TEXT);
INSERT INTO "Stage_Play" VALUES(1,'Hamlet',1990,'Tragedy','The Royal Shakespeare Company','Julia Roberts');
INSERT INTO "Stage_Play" VALUES(2,'The Glass Menagerie',1945,'Drama','Broadway Theatre Group','Peter Brook');
COMMIT;
