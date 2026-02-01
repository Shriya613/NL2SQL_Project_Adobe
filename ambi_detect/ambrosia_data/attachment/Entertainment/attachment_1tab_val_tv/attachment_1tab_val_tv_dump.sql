BEGIN TRANSACTION;
CREATE TABLE CastMember (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    first_name TEXT,
    last_name TEXT,
    birth_date DATE);
INSERT INTO "CastMember" VALUES(1,'Peter','Coyote','1946-08-11');
INSERT INTO "CastMember" VALUES(2,'Elizabeth','Banks','1974-02-10');
INSERT INTO "CastMember" VALUES(3,'Kit','Harrington','1986-06-26');
INSERT INTO "CastMember" VALUES(4,'Emilia','Clarke','1986-10-16');
INSERT INTO "CastMember" VALUES(5,'Jason','Momoa','1979-08-01');
INSERT INTO "CastMember" VALUES(6,'Lena','Headey','1973-10-03');
CREATE TABLE Genre (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT UNIQUE);
INSERT INTO "Genre" VALUES(1,'Action');
INSERT INTO "Genre" VALUES(2,'Comedy');
INSERT INTO "Genre" VALUES(3,'Drama');
INSERT INTO "Genre" VALUES(4,'Thriller');
INSERT INTO "Genre" VALUES(5,'Documentary');
INSERT INTO "Genre" VALUES(6,'Crime');
INSERT INTO "Genre" VALUES(7,'Fantasy');
CREATE TABLE ShowCast (
    show_id INTEGER,
    member_id INTEGER,
    role TEXT,
    appearance_start_year INTEGER,
    appearance_end_year INTEGER,
    FOREIGN KEY(show_id) REFERENCES TvShow(id),
    FOREIGN KEY(member_id) REFERENCES CastMember(id));
INSERT INTO "ShowCast" VALUES(1,1,'Narrator',2006,NULL);
INSERT INTO "ShowCast" VALUES(2,2,'DCI Gillian Bray',2008,2013);
INSERT INTO "ShowCast" VALUES(3,3,'Ned Stark',2011,2014);
INSERT INTO "ShowCast" VALUES(3,4,'Daenerys Targaryen',2011,2019);
INSERT INTO "ShowCast" VALUES(5,5,'Walter White',2008,2012);
INSERT INTO "ShowCast" VALUES(6,6,'Geralt of Rivia',2019,NULL);
CREATE TABLE TvShow (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT,
    tv VARCHAR(50) ,
    number_of_episodes INT,
    release_date DATE,
    genre_id INTEGER,
    FOREIGN KEY(genre_id) REFERENCES Genre(id));
INSERT INTO "TvShow" VALUES(1,'Planet Earth','Documentary Series',8,'2006-03-29',1);
INSERT INTO "TvShow" VALUES(2,'Human Planet','Documentary Series',7,'2009-01-09',2);
INSERT INTO "TvShow" VALUES(3,'Game of Thrones','Mini-Series',8,'2011-04-17',3);
INSERT INTO "TvShow" VALUES(4,'Big Little Lies','Mini-Series',7,'2017-02-19',4);
INSERT INTO "TvShow" VALUES(5,'Breaking Bad','Crime Drama',62,'2008-01-20',5);
INSERT INTO "TvShow" VALUES(6,'The Witcher','Fantasy',16,'2019-12-20',6);
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Genre',7);
INSERT INTO "sqlite_sequence" VALUES('TvShow',6);
INSERT INTO "sqlite_sequence" VALUES('CastMember',6);
COMMIT;
