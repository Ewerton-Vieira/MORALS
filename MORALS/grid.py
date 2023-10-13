# Grid.py  # 2022-11-22
# MIT LICENSE 2020 Ewerton R. Vieira
import numpy as np
import csv

class Grid:
    def __init__(self, lower_bounds, upper_bounds, subdivision, base_name=""):
        self.lower_bounds = lower_bounds
        self.upper_bounds = upper_bounds
        self.subdivision = subdivision
        self.base_name = base_name
        self.dim = len(lower_bounds)
        self.subdiv = self.coordinate_subdiv()

        #size of the box for the uniform initial subdivision
        self.size_of_box = [(upper_bounds[a] - lower_bounds[a])/(2**self.subdiv[a]) for a in range(self.dim)]

    def coordinate_subdiv(self):
        base_sub = self.subdivision // self.dim  # equal subdivision for all variables
        subdiv = [base_sub for i in range(self.dim)]
        remainder = self.subdivision % self.dim
        for i in range(remainder):  # adding extra 1 from each variables when dim doesnt subdivide subdivision
            subdiv[i] += 1
        return subdiv

    def coordinates2index(self, coordinate):
        """Given cell coordinates return the index of a
        cell based on CMGDB. Only works for uniform grid"""
        remainder = self.subdivision % self.dim
        added_remainder = [(remainder + self.dim - 1 - a) % self.dim for a in range(self.dim)]
        sum = 0
        for i in range(self.dim):
            c = coordinate[i]
            count = 0
            while c:
                sum += (c & 1) * 2 ** (added_remainder[i] + count*self.dim)
                c = c >> 1
                count +=1
        return sum

    def point2cell_coord(self, point):
        """Return coordinate of cell on grid that contains the input.
        Caution:coordinate of cell is equal to coordinate of vertex on grid for region_of_cube 0"""

        coordinate = []
        for i in range(self.dim):

            coordinate_temp_i = int((point[i] - self.lower_bounds[i]) / self.size_of_box[i])

            # update coordinate in case it falls on the right fringe (right boundary)
            if coordinate_temp_i == 2**self.subdiv[i]:
                coordinate_temp_i -= 1
            coordinate.append(coordinate_temp_i)

        return coordinate

    def point2cell(self, point):
        """Return the cell id on grid that contains the input point
        Caution:id of cell is equal to id of vertex on grid for region_of_cube 0"""
        coordinate = self.point2cell_coord(point)
        return self.get_id_vertex(coordinate, self.dim, self.subdivision)

    def write_map_grid(self, f, base_name="", write_regions=False):
        """write a csv file with image of all vertex of a grid
        given by lower_bounds, upper_bounds, subdivision
        Input:
        f : funtion
        lower_bounds, upper_bounds, subdivision : information about the cubical grid
        base_name : name of the output file
        write_regions : add lines with regions information (useful to check by hand)"""

        base_name += ".csv"
        with open(base_name, 'w') as file_:
            file = csv.writer(file_)
            file.writerow([self.lower_bounds, self.upper_bounds, self.subdivision])

            for region_index in range(2 ** self.dim):

                if write_regions:
                    region_of_cube = f"{region_index:b}"
                    file.writerow([region_of_cube])

                sub_subdivision, dim_face = self.subdivision_of_face(region_index)

                # print(region_index, sub_subdivision)
                for id in range(2**sub_subdivision):

                    position = self.position_at_grid(id, dim_face, sub_subdivision)
                    x = self.grid_vertex2vertex(region_index, position)

                    # x = position
                    file.writerow(f(x))

    def grid_vertex2vertex(self, face, position):

        count = count_for_face = 0
        x = []
        while count < self.dim:
            if (face & 1):
                x.append(self.upper_bounds[count])
                count_for_face -= 1
            else:
                x.append(self.lower_bounds[count] + self.size_of_box[count]*position[count_for_face])
            face >>= 1
            count += 1
            count_for_face += 1

        return x

    def subdivision_of_face(self, face):
        # subdivision on a face (example face = 0101 has 2**2 subdivisions)
        count = sub_subdivision = dim_face = 0
        face = 2**self.dim + ~face  # flip bits with cap size with 2**dim
        while face:
            if (face & 1):
                sub_subdivision += self.subdiv[count]
                dim_face += 1
            face >>= 1
            count += 1

        return sub_subdivision, dim_face

    def position_at_grid(self, id, dim, subdivision):
        """Given a number (id of a vertice), return the coordinates"""
        position = [0 for i in range(dim)] #[0] * dim
        index = subdivision

        if id >= 2**subdivision:
            return "invalide id for vertice"

        count = 0
        while id or index:
            position[count % dim] <<= 1
            position[count % dim] += (id & 1)
            count += 1
            id >>= 1
            index -= 1
        return position

    def get_id_vertex(self, position, dim, subdivision):
        """Given the position of a vertex, return the id, inverse of position_at_grid"""
        # if len(position) != dim:
        #     return "invalide position or/and dimension"
        position_temp = list(position)
        count = subdivision
        id = 0
        while any(position_temp) or count:
            id <<= 1
            j = (count - 1) % dim
            id += position_temp[j] & 1
            position_temp[j] >>= 1
            count -= 1
        return id

    def vertex2grid_vertex(self, vertex):
        """Return the region_of_cube and the coordinates of a point (geometric vertex) at grid"""

        coordinate = [int(
            np.rint((vertex[a] - self.lower_bounds[a]) / self.size_of_box[a])
        ) for a in range(self.dim)
        ]
        # print(f"coordinate in the cube {coordinate}")

        region_of_cube = [coordinate[a] == 2**self.subdiv[a] for a in range(self.dim)]

        # print(region_of_cube)

        coordinate = [coordinate[a] - region_of_cube[a]
                      for a in range(self.dim) if not region_of_cube[a]]  # update coordinate for the region_of_cube

        if not coordinate:
            coordinate = [0]

        # print(f"coordinate in the region {coordinate}")

        region_of_cube_bit_form = 0  # from list to bit [1,0,1] to 5
        for index, value in enumerate(region_of_cube):
            region_of_cube_bit_form += (value << index)

        # print(region_of_cube, region_of_cube_bit_form, coordinate)

        sub_subdivision, dim_face = self.subdivision_of_face(region_of_cube_bit_form)

        return region_of_cube_bit_form, self.get_id_vertex(coordinate, dim_face, sub_subdivision)

    def load_map_grid(self, file_name):
        map = []
        with open(file_name, 'r') as file_:
            next(file_)
            file = csv.reader(file_, quoting=csv.QUOTE_NONNUMERIC)
            for row in file:
                map.append(row)
        return map

    def image_of_vertex_from_loaded_map(self, map, vertex):
        """input:
        subdivision can be smaller than subdivision used to compute map
        TODO: for now subdivision is the same to compute map (fix this)
        """
        region_of_cube, id = self.vertex2grid_vertex(vertex)

        # print(region_of_cube)

        region_position_file = 0
        for i in range(region_of_cube):
            # if count:
            num = 0
            for index, value in enumerate(self.bit_to_list(i, self.dim)):
                num += self.subdiv[index] * (not value)
                # not value since 0 counts as interior and it should take in account
            region_position_file += 1 << num

        # print("region and id", region_of_cube, id)
        # print("region_position_file + id", region_position_file + id)
        return map[region_position_file + id]

    def list_to_bit(self):
        return None

    def bit_to_list(self, number, dim):
        return [(number >> i) & 1 for i in range(dim)]

    def id2image(self, data):
        """Input: data (x,f(x))
        Return a list indexed by the cell id on CMGDB where each element
        of the returned list correspond to a list of images f(x)
        for all x in cell id \cap data"""
        id2image_ = [list() for i in range(2 ** self.subdivision)]
        data_x = data[:, 0:self.dim].reshape(-1,self.dim)
        data_fx = data[:, self.dim::].reshape(-1,self.dim)

        for i in range(len(data_x)):
            id2image_[self.point2cell(data_x[i])].append(data_fx[i])

        return id2image_


    def valid_grid(self, data, transform = lambda x : x, neighbors=True):
        """initializing valid list of cell on the grid, it is indexed by the
        number assiging to each cell on the grid in CMGDB"""
        valid_list = [False]*(2**self.subdivision)

        if neighbors:
            for i in range(len(data)):
                z = transform(data[i]).tolist()
                all_neighbors_position = self.neighbors(self.point2cell(z))
                for neighbor_position in all_neighbors_position:
                    # valid_list[self.point2cell(neighbor_position)] = True
                    valid_list[self.get_id_vertex(neighbor_position, self.dim, self.subdivision)] = True
        else:
            for i in range(len(data)):
                z = transform(data[i]).tolist()
                valid_list[self.point2cell(z)] = True
        return valid_list

    def uniform_sample_old(self):
        """Return uniform samples on a given grid"""

        number_extra_subdivisions = 0 # add one extra subdivision for each variable
        size_subdivision = self.subdivision + number_extra_subdivisions*self.dim
        size_of_box = [(self.upper_bounds[a] - self.lower_bounds[a])/(2**(self.subdiv[a]+number_extra_subdivisions)) for a in range(self.dim)]

        data = []
        for i in range(2**(size_subdivision)):
            coordinate = self.position_at_grid(i, self.dim, size_subdivision)
            values = [self.lower_bounds[a] + size_of_box[a] * coordinate[a] for a in range(self.dim)]
            data.append(values)

        data = np.array(data)
        return data


    def uniform_sample(self):
        """Return uniform samples on a given grid"""
        data = []
        for region_index in range(2 ** self.dim):

            sub_subdivision, dim_face = self.subdivision_of_face(region_index)
            for id in range(2**sub_subdivision):
                position = self.position_at_grid(id, dim_face, sub_subdivision)
                x = self.grid_vertex2vertex(region_index, position)
                data.append(x)
        return np.array(data)
    
    def point2indexCMGDB(self, point):
        """Given a point return the index (id) in the CMGDB"""
        cell = self.point2cell(point)
        return self.coordinates2index(self.position_at_grid(cell, self.dim, self.subdivision))

    def neighbors(self, cell_index):
        """Given index of region 0, return the neighbors in region 0 (squared ball with radius 1)"""
        coordinate = self.position_at_grid(cell_index, self.dim, self.subdivision) 

        add_elements = [[0] for i in range(self.dim)]

        vector_limit = []

        total_add_elements = 1
        for i in range(self.dim):
            if coordinate[i] == 2**self.subdiv[i] - 1:  # only in region 0
                add_elements[i]+=[-1]
                total_add_elements *= 2
                vector_limit += [2]
            elif coordinate[i] == 0:
                add_elements[i]+=[1]
                total_add_elements *= 2
                vector_limit += [2]
            else:
                add_elements[i]+=[-1,1]
                total_add_elements *= 3
                vector_limit += [3]

        all_neighbors_position = []

        index = 0
        while index < total_add_elements:
            new_coordinate = []
            quocient = index
            for i in range(self.dim):  #index to vector in \oplus_i Z_vector_limit[i]
                reminder = quocient % vector_limit[i]
                quocient = quocient // vector_limit[i]
                new_coordinate += [coordinate[i] + add_elements[i][reminder]]
            all_neighbors_position += [new_coordinate]
            index += 1
        
        return all_neighbors_position