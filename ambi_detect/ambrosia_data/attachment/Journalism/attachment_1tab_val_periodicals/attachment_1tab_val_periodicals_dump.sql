BEGIN TRANSACTION;
CREATE TABLE Articles (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT,
    content TEXT,
    pub_date DATE,
    periodical_id INTEGER,
    author_id INTEGER,
    FOREIGN KEY (periodical_id) REFERENCES Periodicals(id),
    FOREIGN KEY (author_id) REFERENCES Authors(id));
INSERT INTO "Articles" VALUES(1,'A Tale of Two Cities','This novel by Charles Dickens tells...','2023-01-01',1,2);
INSERT INTO "Articles" VALUES(2,'Pride and Prejudice','This classic love story by Jane Austen...','2023-01-02',2,1);
INSERT INTO "Articles" VALUES(3,'Great Expectations','Another masterpiece by Charles Dickens...','2023-01-03',3,3);
INSERT INTO "Articles" VALUES(4,'War and Peace','Tolstoy''s epic tale of love during war...','2023-01-04',4,6);
INSERT INTO "Articles" VALUES(5,'To Kill a Mockingbird','Harper Lee''s Pulitzer Prize winning book...','2023-01-05',5,7);
INSERT INTO "Articles" VALUES(6,'Moby Dick','Classic adventure novel by Herman Melville.','2023-01-06',6,8);
INSERT INTO "Articles" VALUES(7,'1984','Orwell''s dystopian vision of a totalitarian society.','2023-01-07',7,9);
INSERT INTO "Articles" VALUES(8,'Crime and Punishment','Dostoevsky''s psychological thriller about guilt and redemption.','2023-01-08',8,10);
INSERT INTO "Articles" VALUES(9,'The Great Gatsby','Scott Fitzgerald''s timeless portrayal of the American Dream.','2023-01-09',9,1);
INSERT INTO "Articles" VALUES(10,'Animal Farm','George Orwell''s allegory on political corruption.','2023-01-10',10,2);
CREATE TABLE Authors (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT);
INSERT INTO "Authors" VALUES(1,'Jane Austen');
INSERT INTO "Authors" VALUES(2,'Charles Dickens');
INSERT INTO "Authors" VALUES(3,'Mark Twain');
INSERT INTO "Authors" VALUES(4,'George Orwell');
INSERT INTO "Authors" VALUES(5,'Fyodor Dostoevsky');
INSERT INTO "Authors" VALUES(6,'Leo Tolstoy');
INSERT INTO "Authors" VALUES(7,'Harper Lee');
INSERT INTO "Authors" VALUES(8,'Herman Melville');
INSERT INTO "Authors" VALUES(9,'Emily Bronte');
INSERT INTO "Authors" VALUES(10,'William Shakespeare');
CREATE TABLE Periodicals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    type TEXT ,
    circulation_numbers INTEGER,
    pub_date DATE,
    publisher_id INTEGER,
    FOREIGN KEY (publisher_id) REFERENCES Publishers(id));
INSERT INTO "Periodicals" VALUES(1,'The New York Times','Newspaper',20000,'2023-01-01',1);
INSERT INTO "Periodicals" VALUES(2,'USA Today','Newspaper',18000,'2023-01-02',1);
INSERT INTO "Periodicals" VALUES(3,'Time Magazine','Magazine',20000,'2023-01-03',2);
INSERT INTO "Periodicals" VALUES(4,'National Geographic','Magazine',19000,'2023-01-04',3);
INSERT INTO "Periodicals" VALUES(5,'Wired','Magazine',17000,'2023-01-05',4);
INSERT INTO "Periodicals" VALUES(6,'Forbes','Newspaper',15000,'2023-01-06',5);
INSERT INTO "Periodicals" VALUES(7,'Vanity Fair','Magazine',16000,'2023-01-07',6);
CREATE TABLE Publishers (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT);
INSERT INTO "Publishers" VALUES(1,'McClatchy Company');
INSERT INTO "Publishers" VALUES(2,'Gannett Company');
INSERT INTO "Publishers" VALUES(3,'Meredith Corporation');
INSERT INTO "Publishers" VALUES(4,'Conde Nast');
INSERT INTO "Publishers" VALUES(5,'Hearst Communications');
INSERT INTO "Publishers" VALUES(6,'Rupert Murdoch''s News Corporation');
INSERT INTO "Publishers" VALUES(7,'Time Inc.');
INSERT INTO "Publishers" VALUES(8,'Newsweek Media Group');
INSERT INTO "Publishers" VALUES(9,'The Washington Post');
INSERT INTO "Publishers" VALUES(10,'Dow Jones & Company');
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Periodicals',7);
INSERT INTO "sqlite_sequence" VALUES('Authors',10);
INSERT INTO "sqlite_sequence" VALUES('Publishers',10);
INSERT INTO "sqlite_sequence" VALUES('Articles',10);
COMMIT;
