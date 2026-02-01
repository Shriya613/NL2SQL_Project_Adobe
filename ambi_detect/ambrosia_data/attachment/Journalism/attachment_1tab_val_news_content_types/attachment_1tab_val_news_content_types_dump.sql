BEGIN TRANSACTION;
CREATE TABLE Article (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT,
    content TEXT,
    news_content_type TEXT ,
    volume TEXT,
    date DATETIME DEFAULT CURRENT_TIMESTAMP,
    author_id INTEGER,
    category_id INTEGER,
    publication_id INTEGER,
    FOREIGN KEY (author_id) REFERENCES Author(id),
    FOREIGN KEY (category_id) REFERENCES Category(id),
    FOREIGN KEY (publication_id) REFERENCES Publication(id));
INSERT INTO "Article" VALUES(1,'Opinion on Brexit','This is a piece about...','Opinion Piece','01','2024-03-07 23:50:16',1,1,1);
INSERT INTO "Article" VALUES(2,'Climate Change Solutions','A discussion about...','Opinion Piece','02','2024-03-07 23:50:16',2,3,2);
INSERT INTO "Article" VALUES(3,'Apple Launches iPhone 12','Details of Apple’s new phone.','Feature Article','01','2024-03-07 23:50:16',3,4,3);
INSERT INTO "Article" VALUES(4,'Tesla’s Solar Roof','Innovations by Tesla in solar energy.','Feature Article','03','2024-03-07 23:50:16',4,5,4);
INSERT INTO "Article" VALUES(5,'Stock Market Rises','Details of today’s stock market rise.','News','01','2024-03-07 23:50:16',5,2,5);
INSERT INTO "Article" VALUES(6,'Football Game Results','Results of last night’s football game.','Sports','02','2024-03-07 23:50:16',1,6,1);
CREATE TABLE Author (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    email TEXT UNIQUE);
INSERT INTO "Author" VALUES(1,'Alice Smith','alice@example.com');
INSERT INTO "Author" VALUES(2,'Bob Johnson','bob@example.com');
INSERT INTO "Author" VALUES(3,'Charlie Brown','charlie@example.com');
INSERT INTO "Author" VALUES(4,'David Lee','david@example.com');
INSERT INTO "Author" VALUES(5,'Eva Green','eva@example.com');
CREATE TABLE Category (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT,
    description TEXT);
INSERT INTO "Category" VALUES(1,'Politics','Latest political events.');
INSERT INTO "Category" VALUES(2,'Culture','Cultural happenings around the world.');
INSERT INTO "Category" VALUES(3,'Technology','The latest tech trends.');
INSERT INTO "Category" VALUES(4,'Business','Global business news.');
INSERT INTO "Category" VALUES(5,'Entertainment','Hollywood gossip and entertainment news.');
CREATE TABLE Publication (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    address TEXT);
INSERT INTO "Publication" VALUES(1,'New York Times','229 W 43rd St, New York, NY 10036, USA');
INSERT INTO "Publication" VALUES(2,'Washington Post','1301 K Street NW, Washington, DC 20071, USA');
INSERT INTO "Publication" VALUES(3,'Wall Street Journal','One Market Plaza, San Francisco, CA 94105, USA');
INSERT INTO "Publication" VALUES(4,'Los Angeles Times','202 W 1st St, Los Angeles, CA 90012, USA');
INSERT INTO "Publication" VALUES(5,'USA Today','1 Tower Center Blvd, Secaucus, NJ 07094, USA');
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Author',5);
INSERT INTO "sqlite_sequence" VALUES('Category',5);
INSERT INTO "sqlite_sequence" VALUES('Publication',5);
INSERT INTO "sqlite_sequence" VALUES('Article',6);
COMMIT;
