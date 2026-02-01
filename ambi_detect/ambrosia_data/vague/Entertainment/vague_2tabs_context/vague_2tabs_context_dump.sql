BEGIN TRANSACTION;
CREATE TABLE `Author` (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    first_name TEXT,
    last_name TEXT
);
INSERT INTO "Author" VALUES(1,'William','Shakespeare');
INSERT INTO "Author" VALUES(2,'George','Orwell');
INSERT INTO "Author" VALUES(3,'Jane','Austen');
INSERT INTO "Author" VALUES(4,'Fyodor','Dostoevsky');
INSERT INTO "Author" VALUES(5,'Ernest','Hemingway');
CREATE TABLE `Epoch` (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    description TEXT
);
INSERT INTO "Epoch" VALUES(1,'Ancient Greece','A civilization known for its contributions to philosophy and arts.');
INSERT INTO "Epoch" VALUES(2,'Renaissance Europe','A period of great artistic and intellectual achievement.');
INSERT INTO "Epoch" VALUES(3,'Victorian Era','Characterized by rapid industrialization and social change.');
INSERT INTO "Epoch" VALUES(4,'20th Century America','A time of significant cultural shifts and technological advancements.');
INSERT INTO "Epoch" VALUES(5,'Modern Times','The current era marked by global interconnectedness and digital revolution.');
CREATE TABLE `Genre` (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    genre_type TEXT UNIQUE
);
INSERT INTO "Genre" VALUES(1,'Tragedy');
INSERT INTO "Genre" VALUES(2,'Comedy');
INSERT INTO "Genre" VALUES(3,'Mystery');
INSERT INTO "Genre" VALUES(4,'Romance');
INSERT INTO "Genre" VALUES(5,'Science Fiction');
CREATE TABLE `Play` (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT,
    summary TEXT,
    year_of_creation INT,
    epoch_id INTEGER,
    society_id INTEGER,
    author_id INTEGER,
    genre_id INTEGER,
    FOREIGN KEY(epoch_id) REFERENCES Epoch(id),
    FOREIGN KEY(society_id) REFERENCES Society(id),
    FOREIGN KEY(author_id) REFERENCES Author(id),
    FOREIGN KEY(genre_id) REFERENCES Genre(id)
);
INSERT INTO "Play" VALUES(1,'Hamlet','Prince of Denmark seeks revenge on his uncle.',1603,2,3,1,1);
INSERT INTO "Play" VALUES(2,'1984','Set in a dystopian future where totalitarianism is rampant.',1949,4,4,2,3);
INSERT INTO "Play" VALUES(3,'Pride and Prejudice','Follows Elizabeth Bennet as she navigates love and marriage.',1813,3,3,3,4);
INSERT INTO "Play" VALUES(4,'Crime and Punishment','A psychological thriller about a student who commits murder.',1866,4,4,4,5);
INSERT INTO "Play" VALUES(5,'Brave New World','Explores a utopian society built on genetic engineering and conditioning.',1932,4,4,5,6);
CREATE TABLE `Society` (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    culture TEXT
);
INSERT INTO "Society" VALUES(1,'Athenian','Known for democracy, philosophy, and literature.');
INSERT INTO "Society" VALUES(2,'Florentine','Home to the Renaissance and a center for art and learning.');
INSERT INTO "Society" VALUES(3,'Elizabethan-Jacobean England','Known for the flourishing of significant developments in literature and the arts.');
INSERT INTO "Society" VALUES(4,'American','Diverse and multicultural with strong individualism.');
INSERT INTO "Society" VALUES(5,'Global Digital','Connected through technology and social media.');
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Epoch',5);
INSERT INTO "sqlite_sequence" VALUES('Society',5);
INSERT INTO "sqlite_sequence" VALUES('Author',5);
INSERT INTO "sqlite_sequence" VALUES('Genre',5);
INSERT INTO "sqlite_sequence" VALUES('Play',5);
COMMIT;
