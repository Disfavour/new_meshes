import gmsh
from triangle_circumcenter import center_circumscribed_circle
#from mesh_generation.triangle_circumcenter import center_circumscribed_circle


def create_meshes(trinagle_mesh, quadrangle_mesh, small_triangle_mesh, small_quadrangle_mesh):
    gmsh.initialize()

    gmsh.open(trinagle_mesh)

    nodeTags, nodeCoords, nodeParams = gmsh.model.mesh.getNodes()

    elementType = gmsh.model.mesh.getElementType('Triangle', 1)
    elementTags, elementNodeTags = gmsh.model.mesh.getElementsByType(elementType)
    edgeNodes = gmsh.model.mesh.getElementEdgeNodes(elementType)

    gmsh.model.mesh.createEdges()

    edgeTags, edgeOrientations = gmsh.model.mesh.getEdges(edgeNodes)

    edges2Elements = {}
    element2edges = {}
    for i in range(len(edgeTags)): # 3 edges per Triangle
        if not edgeTags[i] in edges2Elements:
            edges2Elements[edgeTags[i]] = [elementTags[i // 3]]
            
        else:
            edges2Elements[edgeTags[i]].append(elementTags[i // 3])
        
        if not elementTags[i // 3] in element2edges:
            element2edges[elementTags[i // 3]] = [edgeTags[i]]
            #edges2Elements[edgeTags[i]] = [elementTags[i // 3]]
            
        else:
            #edges2Elements[edgeTags[i]].append(elementTags[i // 3])
            element2edges[elementTags[i // 3]].append(edgeTags[i])
    
    # print(edges2Elements, element2edges)
    # exit()

    elementNodes = elementNodeTags.reshape((-1, 3))
    nodeCoords = nodeCoords.reshape((-1, 3))
    element2center = {}

    maxNodeTag = gmsh.model.mesh.getMaxNodeTag()

    newNodeTags = []
    newNodeCoords = []

    voronoi_volume = {}
    delaunay_volume = {}

    max_angle = 0
    for element, nodes in zip(elementTags, elementNodes):
        center, max_angle_cur = center_circumscribed_circle(*map(lambda x: nodeCoords[x], nodes - 1))

        newNodeTags.append(maxNodeTag + 1)
        newNodeCoords += list(center)

        element2center[element] = maxNodeTag + 1
        max_angle = max(max_angle_cur, max_angle)

        voronoi_volume[maxNodeTag + 1] = nodes

        maxNodeTag += 1

    gmsh.model.mesh.addNodes(2, 1, newNodeTags, newNodeCoords)

    # New quad elements
    maxElementTag = gmsh.model.mesh.getMaxElementTag()

    # unique edges to nodes

    edge2nodes = {}
    for i, edge in enumerate(edgeTags):
        if edge not in edge2nodes:
            edge2nodes[edge] = edgeNodes[i*2:i*2+2]


    boundary_element2node = {}  # середина ребра на границе для вороного

    quadTags = []
    quadNodes = []
    for edge in edge2nodes:#set(edgeTags):
        triangles = edges2Elements[edge]
        curedgeNodes = edge2nodes[edge]

        # грань входит в 2 треугольника
        if len(triangles) == 2:
            centerNodes = [element2center[triangles[0]], element2center[triangles[1]]]
            quadNodes += [curedgeNodes[0], centerNodes[0], curedgeNodes[1], centerNodes[1]]
        
        # грань на границе
        else:
            center = element2center[triangles[0]]
            # add node
            coord1, parametricCoord1, dim1, tag1 = gmsh.model.mesh.getNode(curedgeNodes[0])
            coord2, parametricCoord2, dim2, tag2 = gmsh.model.mesh.getNode(curedgeNodes[1])
            coord = (coord1 + coord2) / 2

            # 2 1 это плоскость, желательно к линиям mb getClosestPoint
            gmsh.model.mesh.addNodes(2, 1, [maxNodeTag + 1], coord)

            boundary_element2node[triangles[0]] = maxNodeTag + 1

            maxNodeTag += 1

            quadNodes += [curedgeNodes[0], center, curedgeNodes[1], maxNodeTag]

        quadTags.append(maxElementTag+1)
        maxElementTag += 1
    
    node2edges = {}
    for edge in edge2nodes:
        nodes = edge2nodes[edge]
        for node in nodes:
            if node not in node2edges:
                node2edges[node] = [edge]
            else:
                node2edges[node].append(edge)

    # через грани
    for node in node2edges:
        edges = node2edges[node]
        delaunay_volume[node] = []

        start_edge = None

        all_elements = set()
        adj_map = {}
        for edge in edges:
            for element in edges2Elements[edge]:
                all_elements.add(element)

                if element in boundary_element2node:
                    start_element = element

                if element not in adj_map:
                    adj_map[element] = []
                adj_map[element].extend(list(set(edges2Elements[edge]).difference(set([element]))))

        # она соединена только с 1 треуголььником
        for edge in edges:
            if len(edges2Elements[edge]) == 1:
                start_edge = edge
                break
        
        if start_edge is not None:
            linked_triangle = edges2Elements[start_edge][0]
            delaunay_volume[node].append(boundary_element2node[linked_triangle])

            #delaunay_volume[node].append(element2center[linked_triangle])

            start_element = linked_triangle
            
            delaunay_volume[node].append(element2center[start_element])

            visited = [start_element]
            prev = start_element
        
            while set(adj_map[prev]).difference(set(visited)):
                cur = list(set(adj_map[prev]).difference(set(visited)))[0]

                delaunay_volume[node].append(element2center[cur])

                prev = cur
                visited.append(prev)
            
            # if prev in boundary_element2node:
            #     delaunay_volume[node].append(boundary_element2node[prev])
            delaunay_volume[node].append(boundary_element2node[prev])
            delaunay_volume[node].append(node)


        else:
            start_edge = edges[0]
            linked_triangle = edges2Elements[start_edge][0]

            start_element = linked_triangle
            
            delaunay_volume[node].append(element2center[start_element])

            visited = [start_element]
            prev = start_element
        
            while set(adj_map[prev]).difference(set(visited)):
                cur = list(set(adj_map[prev]).difference(set(visited)))[0]

                delaunay_volume[node].append(element2center[cur])

                prev = cur
                visited.append(prev)

    quadElement = gmsh.model.mesh.getElementType("Quadrangle", 1)

    gmsh.model.mesh.addElementsByType(1, quadElement, quadTags, quadNodes)

    gmsh.model.mesh.removeElements(2, 1, elementTags)

    gmsh.write(quadrangle_mesh)

    #gmsh.fltk.run()

    #gmsh.model.mesh.removeElements(2, 1, quadTags)
    # small triangles
    #print(elementTags, elementNodes, element2center)

    small_triangles_nodes = []
    small_triangles_tags = []


    for el, nodes in zip(elementTags, elementNodes):
        #print(nodes, nodes[1:] + [nodes[0]], nodes[1:], [nodes[0]])
        for nodes2 in zip(nodes, list(nodes[1:]) + [nodes[0]]):
            small_triangles_tags.append(maxElementTag + 1)

            small_triangles_nodes += nodes2
            small_triangles_nodes.append(element2center[el])

            maxElementTag += 1
            #print(nodes2)
        #exit()

    triangleElement = gmsh.model.mesh.getElementType("Triangle", 1)

    gmsh.model.mesh.addElementsByType(1, triangleElement, small_triangles_tags, small_triangles_nodes)

    gmsh.model.mesh.removeElements(2, 1, quadTags)

    gmsh.write(small_triangle_mesh)
    #exit()

    # small quadrangles
    small_quadrangles_nodes = []
    small_quadrangles_tags = []

    nodeTagsEdgeCenter = []
    nodeCoordsEdgeCenter = []

    edge2center = {}

    # print(edgeTags, len(edgeTags), len(set(edgeTags)))
    # exit()

    for edge in set(edgeTags):
        if len(edges2Elements[edge]) > 0:   # лишняя нода по идее по границам
            nodeTagsEdgeCenter.append(maxNodeTag + 1)
            #nodeCoordsEdgeCenter += list(nodeCoords[edge2nodes[edge] - 1].flatten())

            p1, p2 = nodeCoords[edge2nodes[edge] - 1]

            node = (p1 + p2) / 2

            nodeCoordsEdgeCenter += list(node)

            # print(nodeCoordsEdgeCenter)
            # exit()

            edge2center[edge] = maxNodeTag + 1

            # print(nodeTags)

            # print(edge2nodes[edge], edge2nodes[edge] - 1, nodeCoords[edge2nodes[edge]], nodeCoords[edge2nodes[edge]].flatten())
            # exit()

            maxNodeTag += 1
    
    gmsh.model.mesh.addNodes(2, 1, nodeTagsEdgeCenter, nodeCoordsEdgeCenter)

    for el in elementTags:
        edges = element2edges[el]
        #print(edges)
        for edges2 in zip(edges, list(edges[1:]) + [edges[0]]):
            #print(edges2)
            node = set(edge2nodes[edges2[0]]).intersection(set(edge2nodes[edges2[1]])).pop()

            small_quadrangles_tags.append(maxElementTag + 1)

            small_quadrangles_nodes += [element2center[el], edge2center[edges2[0]], node, edge2center[edges2[1]]]

            maxElementTag += 1
        

    gmsh.model.mesh.addElementsByType(1, quadElement, small_quadrangles_tags, small_quadrangles_nodes)

    #print(small_quadrangles_tags)
    #print(small_quadrangles_nodes)

    gmsh.model.mesh.removeElements(2, 1, small_triangles_tags)

    gmsh.model.mesh.remove_duplicate_nodes()
    
    gmsh.write(small_quadrangle_mesh)
    #exit()

    gmsh.fltk.run()

    gmsh.finalize()

    print(f'Max angle in triangle mesh {max_angle}')


if __name__ == '__main__':
    create_meshes('meshes/new_triangle.msh',
                  quadrangle_mesh='meshes/quadrangle.msh',
                  small_triangle_mesh='meshes/small_triangle.msh',
                  small_quadrangle_mesh='meshes/small_quadrangle.msh')
