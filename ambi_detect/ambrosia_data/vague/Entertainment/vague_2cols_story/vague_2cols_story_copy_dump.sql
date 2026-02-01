BEGIN TRANSACTION;
CREATE TABLE Authors (
    AuthorID INTEGER PRIMARY KEY,
    FirstName TEXT,
    LastName TEXT,
    BirthDate DATE,
    DeathDate DATE,
    Biography TEXT
);
INSERT INTO "Authors" VALUES(1,'Jane','Austen','1775-04-16',NULL,'Jane Austen was an English novelist known primarily for her six major novels...');
INSERT INTO "Authors" VALUES(2,'George','Orwell','1903-06-25','1950-01-21','George Orwell was a British writer and journalist...');
INSERT INTO "Authors" VALUES(3,'Fyodor','Dostoevsky','1821-11-11','1881-02-09','Fyodor Dostoevsky was a Russian author...');
INSERT INTO "Authors" VALUES(4,'Mark','Twain','1835-11-30','1910-04-21','Samuel Langhorne Clemens, better known by his pen name Mark Twain...');
INSERT INTO "Authors" VALUES(5,'William','Shakespeare','1564-04-26','1616-04-23','William Shakespeare was an English poet, playwright...');
CREATE TABLE Book_Authors (
    BookID INTEGER,
    AuthorID INTEGER,
    FOREIGN KEY (BookID) REFERENCES Books(BookID),
    FOREIGN KEY (AuthorID) REFERENCES Authors(AuthorID),
    PRIMARY KEY (BookID, AuthorID)
);
CREATE TABLE Book_Genres (
    BookID INTEGER,
    GenreID INTEGER,
    FOREIGN KEY (BookID) REFERENCES Books(BookID),
    FOREIGN KEY (GenreID) REFERENCES Genres(GenreID),
    PRIMARY KEY (BookID, GenreID)
);
CREATE TABLE "Books" (
    BookID INTEGER PRIMARY KEY,
    Title TEXT,
    Author TEXT,
    PublicationYear INT ,
    Story TEXT,
    Novel BOOLEAN DEFAULT FALSE
);
INSERT INTO "Books" VALUES(1,'Pride and Prejudice','Jane Austen',1797,'The story follows Elizabeth Bennet as she deals with issues of marriage, morality, education...',1);
INSERT INTO "Books" VALUES(2,'1984','George Orwell',1949,'In a dystopian future...',0);
INSERT INTO "Books" VALUES(3,'Crime and Punishment','Fyodor Dostoevsky',1866,'A psychological thriller that delves into the mind of Raskolnikov...',0);
INSERT INTO "Books" VALUES(4,'Adventures of Huckleberry Finn','Mark Twain',1884,'Set against the backdrop of slavery in mid-19th century America...',0);
INSERT INTO "Books" VALUES(5,'Hamlet','William Shakespeare',1603,'Prince Hamlet seeks revenge on his uncle...',1);
CREATE TABLE Genres (
    GenreID INTEGER PRIMARY KEY,
    Name TEXT UNIQUE
);
INSERT INTO "Genres" VALUES(1,'Romance');
INSERT INTO "Genres" VALUES(2,'Mystery');
INSERT INTO "Genres" VALUES(3,'Historical Fiction');
INSERT INTO "Genres" VALUES(4,'Science Fiction');
INSERT INTO "Genres" VALUES(5,'Thriller');
CREATE TABLE Publishers (
    PublisherID INTEGER PRIMARY KEY,
    Name TEXT,
    Address TEXT,
    PhoneNumber VARCHAR(20)
);
INSERT INTO "Publishers" VALUES(1,'Penguin Random House','New York City','+1-212-366-2000');
INSERT INTO "Publishers" VALUES(2,'HarperCollins','New York City','+1-212-207-7000');
INSERT INTO "Publishers" VALUES(3,'Hachette Book Group','New York City','+1-212-364-1000');
INSERT INTO "Publishers" VALUES(4,'Simon & Schuster','New York City','+1-212-698-7000');
INSERT INTO "Publishers" VALUES(5,'Macmillan Publishing','New York City','+1-212-339-6000');
COMMIT;
