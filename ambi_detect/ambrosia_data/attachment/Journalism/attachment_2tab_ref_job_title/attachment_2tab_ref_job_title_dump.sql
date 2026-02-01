BEGIN TRANSACTION;
CREATE TABLE Articles(
    id INTEGER PRIMARY KEY,
    title TEXT,
    content TEXT,
    published_date DATETIME DEFAULT CURRENT_TIMESTAMP,
    author_id INTEGER REFERENCES Authors(id));
INSERT INTO "Articles" VALUES(1,'Article 1','Content for Article 1...','2022-01-01 10:00:00',1);
INSERT INTO "Articles" VALUES(2,'Article 2','Content for Article 2...','2022-01-02 10:00:00',2);
CREATE TABLE Authors(
    id INTEGER PRIMARY KEY,
    name TEXT,
    organization TEXT,
    public_relations_contact_id INTEGER REFERENCES Contacts(public_relations_id),
    government_communications_contact_id INTEGER REFERENCES Contacts(government_communications_id));
INSERT INTO "Authors" VALUES(1,'Michael Johnson','The Guardian',1,1);
INSERT INTO "Authors" VALUES(2,'Emma Davis','The New York Times',2,2);
CREATE TABLE Contacts(
    id INTEGER PRIMARY KEY,
    name TEXT,
    organization TEXT,
    public_relations_id INTEGER REFERENCES Public_Relations(id),
    government_communications_id INTEGER REFERENCES Government_Communications(id));
INSERT INTO "Contacts" VALUES(1,'John Doe','ABC News',1,NULL);
INSERT INTO "Contacts" VALUES(2,'Jane Smith','Fox News',2,1);
INSERT INTO "Contacts" VALUES(3,'Alice Brown','NBC News',NULL,2);
CREATE TABLE Government_Communications(
    id INTEGER PRIMARY KEY,
    contact TEXT,
    department TEXT,
    role TEXT,
    email TEXT,
    phone TEXT,
    description TEXT);
INSERT INTO "Government_Communications" VALUES(1,'WhiteHouse','Executive Branch','Press Secretary','whitehouse@email.gov','555-123-4567','Spokesperson for the White House');
INSERT INTO "Government_Communications" VALUES(2,'StateDept','Executive Branch','Deputy Spokesman','statedept@email.gov','555-987-6543','Assists the Press Secretary at State Department');
CREATE TABLE Public_Relations(
    id INTEGER PRIMARY KEY,
    contact TEXT,
    department TEXT,
    role TEXT,
    email TEXT,
    phone TEXT,
    description TEXT);
INSERT INTO "Public_Relations" VALUES(1,'BBC','News Agency','PR Manager','bbc@email.com','123-456-7890','Manages PR for BBC News');
INSERT INTO "Public_Relations" VALUES(2,'CNN','News Agency','Director of Communications','cnn@email.com','987-654-3210','Oversees communications strategy for CNN');
CREATE TABLE Sources(
    id INTEGER PRIMARY KEY,
    name TEXT,
    type TEXT,
    source_url TEXT,
    article_id INTEGER REFERENCES Articles(id));
INSERT INTO "Sources" VALUES(1,'Source 1','Website','https://www.source1.com/article',1);
INSERT INTO "Sources" VALUES(2,'Source 2','Newspaper','http://newspaper.com/article',2);
COMMIT;
