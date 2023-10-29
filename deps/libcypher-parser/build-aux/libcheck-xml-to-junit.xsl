<?xml version="1.0" encoding="UTF-8"?>
<xsl:stylesheet version="1.0" xmlns:src="http://check.sourceforge.net/ns" xmlns:xsl="http://www.w3.org/1999/XSL/Transform" xmlns:fo="http://www.w3.org/1999/XSL/Format">
    <xsl:output indent="yes"/>
    <xsl:template match="/src:testsuites">
        <xsl:element name="testsuites">
            <xsl:apply-templates select="src:suite"/>
        </xsl:element>
    </xsl:template>
    <xsl:template match="src:suite">
        <xsl:element name="testsuite">
            <xsl:attribute name="failures"><xsl:value-of select="count(src:test[@result='failure'])"/></xsl:attribute>
            <xsl:attribute name="errors">0</xsl:attribute>
            <xsl:attribute name="tests"><xsl:value-of select="count(src:test)"/></xsl:attribute>
            <xsl:attribute name="name"><xsl:value-of select="src:title"/></xsl:attribute>
            <xsl:attribute name="timestamp"><xsl:value-of select="../src:datetime"/></xsl:attribute>
            <xsl:attribute name="time"><xsl:value-of select="../src:duration"/></xsl:attribute>
            <xsl:apply-templates select="src:test"/>
        </xsl:element>
    </xsl:template>
    <xsl:template match="src:test">
        <xsl:element name="testcase">
            <xsl:attribute name="name"><xsl:value-of select="src:id"/></xsl:attribute>
            <xsl:attribute name="time"><xsl:value-of select="src:duration"/></xsl:attribute>
            <xsl:if test="@result='failure'">
                <xsl:call-template name="failure"/>
            </xsl:if>
        </xsl:element>
    </xsl:template>
    <xsl:template name="failure">
        <xsl:element name="failure">
            <xsl:attribute name="message"><xsl:value-of select="src:message"/></xsl:attribute>
            <xsl:value-of select="src:path"/><xsl:text>/</xsl:text><xsl:value-of select="src:fn"/>
        </xsl:element>
    </xsl:template>
</xsl:stylesheet>
