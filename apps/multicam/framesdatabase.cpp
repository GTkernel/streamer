#include "framesdatabase.hpp"
using namespace litesql;
const litesql::FieldType FrameEntry::Own::Id("id_",A_field_type_integer,"FrameEntry_");
const std::string FrameEntry::type__("FrameEntry");
const std::string FrameEntry::table__("FrameEntry_");
const std::string FrameEntry::sequence__("FrameEntry_seq");
const litesql::FieldType FrameEntry::Id("id_",A_field_type_integer,table__);
const litesql::FieldType FrameEntry::Type("type_",A_field_type_string,table__);
const litesql::FieldType FrameEntry::Path("path_",A_field_type_string,table__);
const litesql::FieldType FrameEntry::Date("date_",A_field_type_datetime,table__);
void FrameEntry::initValues() {
}
void FrameEntry::defaults() {
    id = 0;
    date = 0;
}
FrameEntry::FrameEntry(const litesql::Database& db)
     : litesql::Persistent(db), id(Id), type(Type), path(Path), date(Date) {
    defaults();
}
FrameEntry::FrameEntry(const litesql::Database& db, const litesql::Record& rec)
     : litesql::Persistent(db, rec), id(Id), type(Type), path(Path), date(Date) {
    defaults();
    size_t size = (rec.size() > 4) ? 4 : rec.size();
    switch(size) {
    case 4: date = convert<const std::string&, litesql::DateTime>(rec[3]);
        date.setModified(false);
    case 3: path = convert<const std::string&, std::string>(rec[2]);
        path.setModified(false);
    case 2: type = convert<const std::string&, std::string>(rec[1]);
        type.setModified(false);
    case 1: id = convert<const std::string&, int>(rec[0]);
        id.setModified(false);
    }
}
FrameEntry::FrameEntry(const FrameEntry& obj)
     : litesql::Persistent(obj), id(obj.id), type(obj.type), path(obj.path), date(obj.date) {
}
const FrameEntry& FrameEntry::operator=(const FrameEntry& obj) {
    if (this != &obj) {
        id = obj.id;
        type = obj.type;
        path = obj.path;
        date = obj.date;
    }
    litesql::Persistent::operator=(obj);
    return *this;
}
std::string FrameEntry::insert(litesql::Record& tables, litesql::Records& fieldRecs, litesql::Records& valueRecs) {
    tables.push_back(table__);
    litesql::Record fields;
    litesql::Record values;
    fields.push_back(id.name());
    values.push_back(id);
    id.setModified(false);
    fields.push_back(type.name());
    values.push_back(type);
    type.setModified(false);
    fields.push_back(path.name());
    values.push_back(path);
    path.setModified(false);
    fields.push_back(date.name());
    values.push_back(date);
    date.setModified(false);
    fieldRecs.push_back(fields);
    valueRecs.push_back(values);
    return litesql::Persistent::insert(tables, fieldRecs, valueRecs, sequence__);
}
void FrameEntry::create() {
    litesql::Record tables;
    litesql::Records fieldRecs;
    litesql::Records valueRecs;
    type = type__;
    std::string newID = insert(tables, fieldRecs, valueRecs);
    if (id == 0)
        id = newID;
}
void FrameEntry::addUpdates(Updates& updates) {
    prepareUpdate(updates, table__);
    updateField(updates, table__, id);
    updateField(updates, table__, type);
    updateField(updates, table__, path);
    updateField(updates, table__, date);
}
void FrameEntry::addIDUpdates(Updates& ) {
}
void FrameEntry::getFieldTypes(std::vector<litesql::FieldType>& ftypes) {
    ftypes.push_back(Id);
    ftypes.push_back(Type);
    ftypes.push_back(Path);
    ftypes.push_back(Date);
}
void FrameEntry::delRecord() {
    deleteFromTable(table__, id);
}
void FrameEntry::delRelations() {
}
void FrameEntry::update() {
    if (!inDatabase) {
        create();
        return;
    }
    Updates updates;
    addUpdates(updates);
    if (id != oldKey) {
        if (!typeIsCorrect()) 
            upcastCopy()->addIDUpdates(updates);
    }
    litesql::Persistent::update(updates);
    oldKey = id;
}
void FrameEntry::del() {
    if (!typeIsCorrect()) {
        const std::unique_ptr<FrameEntry> p(upcastCopy());
        p->delRelations();
        p->onDelete();
        p->delRecord();
    } else {
        delRelations();
        onDelete();
        delRecord();
    }
    inDatabase = false;
}
bool FrameEntry::typeIsCorrect() const {
    return type == type__;
}
std::unique_ptr<FrameEntry> FrameEntry::upcast() const {
    return unique_ptr<FrameEntry>(new FrameEntry(*this));
}
std::unique_ptr<FrameEntry> FrameEntry::upcastCopy() const {
    FrameEntry* np = new FrameEntry(*this);
    np->id = id;
    np->type = type;
    np->path = path;
    np->date = date;
    np->inDatabase = inDatabase;
    return unique_ptr<FrameEntry>(np);
}
std::ostream & operator<<(std::ostream& os, FrameEntry const& o) {
    os << "-------------------------------------" << std::endl;
    os << o.id.name() << " = " << o.id << std::endl;
    os << o.type.name() << " = " << o.type << std::endl;
    os << o.path.name() << " = " << o.path << std::endl;
    os << o.date.name() << " = " << o.date << std::endl;
    os << "-------------------------------------" << std::endl;
    return os;
}
FramesDatabase::FramesDatabase(std::string backendType, std::string connInfo)
     : litesql::Database(backendType, connInfo) {
    initialize();
}
std::vector<litesql::Database::SchemaItem> FramesDatabase::getSchema() const {
    vector<Database::SchemaItem> res;
    string TEXT = backend->getSQLType(A_field_type_string);
    string rowIdType = backend->getRowIDType();
    res.push_back(Database::SchemaItem("schema_","table","CREATE TABLE schema_ (name_ "+TEXT+", type_ "+TEXT+", sql_ "+TEXT+")"));
    if (backend->supportsSequences()) {
        res.push_back(Database::SchemaItem("FrameEntry_seq","sequence",backend->getCreateSequenceSQL("FrameEntry_seq")));
    }
    res.push_back(Database::SchemaItem("FrameEntry_","table","CREATE TABLE FrameEntry_ (id_ " + rowIdType + ",type_ " + backend->getSQLType(A_field_type_string,"") + "" +",path_ " + backend->getSQLType(A_field_type_string,"") + "" +",date_ " + backend->getSQLType(A_field_type_datetime,"") + "" +")"));
    res.push_back(Database::SchemaItem("FrameEntry_id_idx","index","CREATE INDEX FrameEntry_id_idx ON FrameEntry_ (id_)"));
    return res;
}
void FramesDatabase::initialize() {
    static bool initialized = false;
    if (initialized)
        return;
    initialized = true;
}
