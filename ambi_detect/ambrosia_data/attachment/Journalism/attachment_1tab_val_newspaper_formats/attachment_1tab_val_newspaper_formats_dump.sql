BEGIN TRANSACTION;
CREATE TABLE ArticleCategories(
    id INTEGER PRIMARY KEY,
    CategoryName TEXT,
    Description TEXT
);
INSERT INTO "ArticleCategories" VALUES(1,'Politics','News about governmental activities');
INSERT INTO "ArticleCategories" VALUES(2,'Economy','Business news and analysis');
CREATE TABLE Articles(
    id INTEGER PRIMARY KEY,
    Title TEXT,
    Content TEXT,
    PublicationDate DATE,
    NewspaperId INTEGER,
    AuthorId INTEGER,
    CategoryId INTEGER,
    FOREIGN KEY (NewspaperId) REFERENCES Newspapers(id),
    FOREIGN KEY (AuthorId) REFERENCES Authors(id),
    FOREIGN KEY (CategoryId) REFERENCES ArticleCategories(id)
);
INSERT INTO "Articles" VALUES(1,'UK Election Results','Here are the results of the latest UK election.','2021-10-01',2,1,1);
INSERT INTO "Articles" VALUES(2,'Stock Market Rises','The stock market experienced significant gains today.','2021-10-01',3,2,2);
CREATE TABLE Authors(
    id INTEGER PRIMARY KEY,
    FirstName TEXT,
    LastName TEXT,
    Birthdate DATE
);
INSERT INTO "Authors" VALUES(1,'John','Smith','1960-01-01');
INSERT INTO "Authors" VALUES(2,'Jane','Doe','1975-05-01');
CREATE TABLE Distributors(
    id INTEGER PRIMARY KEY,
    Name TEXT,
    Address TEXT,
    PhoneNumber CHAR(20),
    EmailAddress TEXT
);
INSERT INTO "Distributors" VALUES(1,'Sainsbury''s','123 High St','+44 1234567890','distribution@sainburys.com');
INSERT INTO "Distributors" VALUES(2,'Tesco','456 Low St','+44 0987654321','distribution@tesco.com');
CREATE TABLE Newspapers(
    id INTEGER PRIMARY KEY,
    Name TEXT,
    Format VARCHAR(10) ,
    Distributor TEXT,
    FoundedDate DATE,
    FOREIGN KEY (Distributor) REFERENCES Distributors(Name)
);
INSERT INTO "Newspapers" VALUES(1,'The Times','Broadsheet','Sainsbury''s','1785-01-01');
INSERT INTO "Newspapers" VALUES(2,'The Sun','Tabloid','Sainsbury''s','1964-03-17');
INSERT INTO "Newspapers" VALUES(3,'Mail on Sunday','Compact','Tesco','1982-01-01');
INSERT INTO "Newspapers" VALUES(4,'Guardian Unlimited','Berliner','Tesco','1999-01-01');
INSERT INTO "Newspapers" VALUES(5,'Financial Times','Tabloid','Direct delivery','1888-01-01');
CREATE TABLE Readers(
    id INTEGER PRIMARY KEY,
    FullName TEXT,
    Age INT,
    Email TEXT,
    FavouriteNewspaperId INTEGER,
    FOREIGN KEY (FavouriteNewspaperId) REFERENCES Newspapers(id)
);
COMMIT;
